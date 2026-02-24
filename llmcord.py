import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
import httpx
from openai import AsyncOpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def strip_thinking_tags(text: str) -> str:
    """Remove <think> tags and their content from response text."""
    import re
    # Remove <think>...</think> blocks (handles multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_error_message(error: Exception) -> str:
    """Extract human-readable error message from exception."""
    error_str = str(error)
    error_type = type(error).__name__
    
    # Rate limit errors
    if "429" in error_str or error_type == "RateLimitError":
        # Try to extract the provider's message
        if "is temporarily rate-limited" in error_str:
            match = error_str.split("'raw': '")[1].split("'")[0] if "'raw': '" in error_str else None
            if match:
                return f"⚠️ Rate Limited: {match}"
        return "⚠️ Rate Limited: API provider is temporarily rate-limited. Please retry shortly."
    
    # Authentication errors
    if "401" in error_str or "Unauthorized" in error_str:
        return "❌ Authentication Error: Invalid API key or credentials."
    
    # Not found errors
    if "404" in error_str or error_type == "NotFound":
        return "❌ Not Found: The requested resource was not found."
    
    # Forbidden errors
    if "403" in error_str or error_type == "Forbidden":
        return "❌ Forbidden: You don't have permission to access this resource."
    
    # Connection errors
    if "Connection" in error_type or "ECONNREFUSED" in error_str or "ETIMEDOUT" in error_str:
        return "❌ Connection Error: Unable to connect to the API provider. Check your internet or provider status."
    
    # Generic fallback - just use error type and first line of message
    first_line = error_str.split('\n')[0][:100]
    return f"❌ {error_type}: {first_line}"


async def notify_admin_error(error: Exception, context: str = "") -> None:
    """Send error notification to all admins via DM."""
    try:
        admin_ids = config.get("permissions", {}).get("users", {}).get("admin_ids", [])
        if not admin_ids:
            return
        
        error_msg = parse_error_message(error)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        dm_content = f"🤖 **Bot Error Notification**\n"
        dm_content += f"⏰ Time: {timestamp}\n"
        dm_content += f"📝 Context: {context}\n\n"
        dm_content += f"Error: {error_msg}"
        
        for admin_id in admin_ids:
            try:
                admin_user = discord_bot.get_user(admin_id)
                if not admin_user:
                    admin_user = await discord_bot.fetch_user(admin_id)
                
                await admin_user.send(dm_content)
            except (discord.NotFound, discord.Forbidden):
                logging.warning(f"Could not send error notification to admin {admin_id}")
            except Exception as e:
                logging.warning(f"Failed to notify admin {admin_id}: {e}")
    except Exception as e:
        logging.warning(f"Failed to notify admins of error: {e}")


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

scheduler = AsyncIOScheduler()

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True  # Allow sending DMs to users
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.tree.command(name="clear", description="Clear conversation history and cached messages")
async def clear_command(interaction: discord.Interaction) -> None:
    global msg_nodes
    msg_nodes.clear()
    output = "✅ Conversation history cleared. Starting fresh!"
    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))
    logging.info(f"Message cache cleared by user {interaction.user.id}")


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()
    logging.info(f"Synced {len(discord_bot.tree._get_all_commands())} slash commands")
    
    # Start scheduler for periodic tasks
    if not scheduler.running:
        scheduler.start()
        setup_scheduled_tasks()
        logging.info("Scheduler started")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    show_embed_color = config.get("show_embed_color", True)
    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))
        if not show_embed_color:
            embed.color = None

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    model_used = provider_slash_model
    fallback_attempted = False
    
    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        # Strip thinking tags before displaying
                        display_content = strip_thinking_tags(response_contents[-1]) if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.description = display_content
                        
                        if show_embed_color:
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    # Strip thinking tags from plain responses
                    clean_content = strip_thinking_tags(content)
                    if clean_content:
                        await reply_helper(content=clean_content)

    except Exception as e:
        # Try fallback models if configured
        fallback_models = config.get("fallback_models", []) or []
        if fallback_models and not fallback_attempted:
            for fallback_idx, fallback_model in enumerate(fallback_models):
                if not fallback_model or not fallback_model.strip():
                    continue
                    
                fallback_attempted = True
                logging.warning(f"Primary model '{provider_slash_model}' failed: {parse_error_message(e)}. Attempting fallback [{fallback_idx + 1}/{len(fallback_models)}]: {fallback_model}")
                
                try:
                    # Setup fallback client
                    fallback_provider, fallback_model_name = fallback_model.removesuffix(":vision").split("/", 1)
                    fallback_provider_config = config["providers"][fallback_provider]
                    fallback_openai_client = AsyncOpenAI(
                        base_url=fallback_provider_config["base_url"],
                        api_key=fallback_provider_config.get("api_key", "sk-no-key-required")
                    )
                    
                    fallback_model_params = config["models"].get(fallback_model, None)
                    fallback_extra_headers = fallback_provider_config.get("extra_headers")
                    fallback_extra_query = fallback_provider_config.get("extra_query")
                    fallback_extra_body = (fallback_provider_config.get("extra_body") or {}) | (fallback_model_params or {}) or None
                    
                    # Reset for fallback attempt
                    curr_content = finish_reason = None
                    response_contents = []
                    
                    fallback_kwargs = dict(
                        model=fallback_model_name,
                        messages=messages[::-1],
                        stream=True,
                        extra_headers=fallback_extra_headers,
                        extra_query=fallback_extra_query,
                        extra_body=fallback_extra_body
                    )
                    
                    async with new_msg.channel.typing():
                        async for chunk in await fallback_openai_client.chat.completions.create(**fallback_kwargs):
                            if finish_reason != None:
                                break

                            if not (choice := chunk.choices[0] if chunk.choices else None):
                                continue

                            finish_reason = choice.finish_reason
                            prev_content = curr_content or ""
                            curr_content = choice.delta.content or ""
                            new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                            if response_contents == [] and new_content == "":
                                continue

                            if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                                response_contents.append("")

                            response_contents[-1] += new_content

                            if not use_plain_responses:
                                time_delta = datetime.now().timestamp() - last_task_time
                                ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                                msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                                is_final_edit = finish_reason != None or msg_split_incoming
                                is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                                if start_next_msg or ready_to_edit or is_final_edit:
                                    display_content = strip_thinking_tags(response_contents[-1]) if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                                    embed.description = display_content
                                    
                                    if show_embed_color:
                                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                                    if start_next_msg:
                                        await reply_helper(embed=embed, silent=True)
                                    else:
                                        await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                        await response_msgs[-1].edit(embed=embed)

                                    last_task_time = datetime.now().timestamp()

                        if use_plain_responses:
                            for content in response_contents:
                                clean_content = strip_thinking_tags(content)
                                if clean_content:
                                    await reply_helper(content=clean_content)
                    
                    model_used = fallback_model
                    logging.info(f"Fallback model '{fallback_model}' succeeded")
                    break  # Success - exit fallback loop
                    
                except Exception as fallback_error:
                    # This fallback failed, try next one
                    logging.warning(f"Fallback model '{fallback_model}' failed: {parse_error_message(fallback_error)}")
                    if fallback_idx == len(fallback_models) - 1:
                        # Last fallback also failed
                        logging.exception("Error while generating response (all fallback models failed)")
                        channel_info = f"in #{new_msg.channel.name}" if hasattr(new_msg.channel, 'name') else "in DM"
                        await notify_admin_error(fallback_error, f"Failed to generate response {channel_info} (all models failed)")
                    continue
        else:
            # No fallback available or already tried fallback
            logging.exception("Error while generating response")
            channel_info = f"in #{new_msg.channel.name}" if hasattr(new_msg.channel, 'name') else "in DM"
            await notify_admin_error(e, f"Failed to generate response {channel_info}")

    for response_msg in response_msgs:
        # Strip thinking tags before caching
        clean_text = strip_thinking_tags("".join(response_contents))
        msg_nodes[response_msg.id].text = clean_text
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def run_scheduled_task(task_name: str, task_config: dict[str, Any]) -> None:
    """Run a scheduled task with given configuration."""
    try:
        if not task_config.get("enabled", False):
            return
        
        channel_id = task_config.get("channel_id")
        user_id = task_config.get("user_id")
        model_name = task_config.get("model", curr_model)
        prompt = task_config.get("prompt", "Check my emails")
        
        target = None
        
        # Try to get target channel or user
        if channel_id:
            target = discord_bot.get_channel(channel_id)
            if not target:
                logging.warning(f"Scheduled task '{task_name}': channel {channel_id} not found")
                return
        elif user_id:
            # First try to get user from bot's cache (faster and more reliable)
            user_obj = discord_bot.get_user(user_id)
            if not user_obj:
                try:
                    user_obj = await discord_bot.fetch_user(user_id)
                except discord.NotFound:
                    logging.warning(f"Scheduled task '{task_name}': user {user_id} not found")
                    return
            
            # Check if we can DM this user
            if user_obj.bot:
                logging.warning(f"Scheduled task '{task_name}': cannot DM to bots (user {user_id} is a bot)")
                return
            
            target = user_obj
        else:
            logging.warning(f"Scheduled task '{task_name}': no channel_id or user_id configured")
            return
        
        # Setup LLM client
        provider, model = model_name.removesuffix(":vision").split("/", 1)
        provider_config = config["providers"][provider]
        
        base_url = provider_config["base_url"]
        api_key = provider_config.get("api_key", "sk-no-key-required")
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
        model_parameters = config["models"].get(model_name, None)
        extra_headers = provider_config.get("extra_headers")
        extra_query = provider_config.get("extra_query")
        extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None
        
        # Get system prompt if configured
        system_prompt_text = config.get("system_prompt", "")
        if system_prompt_text:
            now = datetime.now().astimezone()
            system_prompt_text = system_prompt_text.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        if system_prompt_text:
            messages.append({"role": "system", "content": system_prompt_text})
        
        
        # Stream response
        response_text = ""
        models_to_try = [(model, openai_client, model_name)]
        
        # Add fallback models if configured (task-specific or global)
        fallback_models = task_config.get("fallback_models") or config.get("fallback_models", []) or []
        
        # Build list of all models to try
        for fallback_model in fallback_models:
            if not fallback_model or not str(fallback_model).strip():
                continue
            try:
                fallback_provider, fallback_model_name = fallback_model.removesuffix(":vision").split("/", 1)
                fallback_provider_config = config["providers"][fallback_provider]
                fallback_openai_client = AsyncOpenAI(
                    base_url=fallback_provider_config["base_url"],
                    api_key=fallback_provider_config.get("api_key", "sk-no-key-required")
                )
                fallback_model_params = config["models"].get(fallback_model, None)
                fallback_extra_headers = fallback_provider_config.get("extra_headers")
                fallback_extra_query = fallback_provider_config.get("extra_query")
                fallback_extra_body = (fallback_provider_config.get("extra_body") or {}) | (fallback_model_params or {}) or None
                
                models_to_try.append((fallback_model_name, fallback_openai_client, fallback_model, fallback_extra_headers, fallback_extra_query, fallback_extra_body))
            except Exception as setup_error:
                logging.warning(f"Scheduled task '{task_name}': Failed to setup fallback model '{fallback_model}': {setup_error}")
        
        last_error = None
        for attempt_idx, model_info in enumerate(models_to_try):
            try:
                response_text = ""
                if attempt_idx == 0:
                    # Primary model
                    attempt_model = model
                    attempt_client = openai_client
                    attempt_model_name = model_name
                    attempt_extra_headers = extra_headers
                    attempt_extra_query = extra_query
                    attempt_extra_body = extra_body
                else:
                    # Fallback model
                    attempt_model, attempt_client, attempt_model_name, attempt_extra_headers, attempt_extra_query, attempt_extra_body = model_info
                
                async for chunk in await attempt_client.chat.completions.create(
                    model=attempt_model,
                    messages=messages[::-1],
                    stream=True,
                    extra_headers=attempt_extra_headers,
                    extra_query=attempt_extra_query,
                    extra_body=attempt_extra_body
                ):
                    if choice := chunk.choices[0] if chunk.choices else None:
                        response_text += choice.delta.content or ""
                
                # Success - break out of retry loop
                if attempt_idx > 0:
                    logging.info(f"Scheduled task '{task_name}': Fallback model [{attempt_idx}/{len(models_to_try)-1}] '{attempt_model_name}' succeeded")
                break
                
            except Exception as e:
                last_error = e
                if attempt_idx == 0:
                    # Primary model failed
                    logging.warning(f"Scheduled task '{task_name}': Primary model failed: {parse_error_message(e)}")
                    if len(models_to_try) > 1:
                        logging.info(f"Scheduled task '{task_name}': Attempting {len(models_to_try) - 1} fallback model(s)")
                    else:
                        # No fallback
                        raise
                else:
                    # Fallback failed
                    fallback_num = attempt_idx
                    total_fallbacks = len(models_to_try) - 1
                    logging.warning(f"Scheduled task '{task_name}': Fallback model [{fallback_num}/{total_fallbacks}] failed: {parse_error_message(e)}")
                    if attempt_idx < len(models_to_try) - 1:
                        # More fallbacks to try
                        continue
                    else:
                        # Last model failed
                        logging.error(f"Scheduled task '{task_name}': All models exhausted")
                        raise
        
        # Send response in chunks if needed
        max_message_length = 4096
        if response_text:
            for i in range(0, len(response_text), max_message_length):
                chunk = response_text[i:i+max_message_length]
                try:
                    await target.send(chunk)
                except discord.Forbidden:
                    logging.error(
                        f"Scheduled task '{task_name}': Cannot DM user {user_id}. "
                        f"The user may have:\n"
                        f"  - DMs disabled from server members/bots\n"
                        f"  - Blocked the bot\n"
                        f"  - Privacy settings that block the bot\n"
                        f"Try having the user start a DM with the bot first, then retry."
                    )
                    return
            target_info = f"channel {channel_id}" if channel_id else f"user {user_id}"
            logging.info(f"Scheduled task '{task_name}' executed: sent results to {target_info}")
        else:
            await target.send(f"📧 No response from task '{task_name}'")
            
    except discord.errors.HTTPException as e:
        if e.code == 50007:  # Cannot send messages to this user
            logging.error(
                f"Scheduled task '{task_name}': Discord error 50007 - Cannot send DM. "
                f"The user may have DMs disabled or has blocked the bot. "
                f"User {user_id} needs to allow DMs from bots in Discord settings."
            )
        else:
            logging.exception(f"Discord error in scheduled task '{task_name}'")
            # Notify admin about Discord errors even if fallback was successful
            await notify_admin_error(e, f"Scheduled task '{task_name}' - Discord error")
    except Exception as e:
        logging.exception(f"Error in scheduled task '{task_name}'")
        # Only notify if all models failed
        await notify_admin_error(e, f"Scheduled task '{task_name}' failed (all {len(models_to_try)} model(s) failed)")


def parse_cron(cron_expr: str) -> dict[str, Any]:
    """Parse cron expression into APScheduler kwargs.
    Format: minute hour day month day_of_week
    Example: '0 9 * * *' = 9:00 AM every day
    Example: '* * * * *' = Every minute
    """
    parts = cron_expr.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron format: {cron_expr}. Use: minute hour day month day_of_week")
    
    minute, hour, day, month, day_of_week = parts
    
    # Always set second=0 to prevent running every second
    kwargs = {"second": 0}
    
    if minute != "*":
        kwargs["minute"] = minute
    if hour != "*":
        kwargs["hour"] = hour
    if day != "*":
        kwargs["day"] = day
    if month != "*":
        kwargs["month"] = month
    if day_of_week != "*":
        kwargs["day_of_week"] = day_of_week
    
    return kwargs


def setup_scheduled_tasks() -> None:
    """Setup scheduled tasks based on config."""
    scheduled_tasks = config.get("scheduled_tasks", {})
    
    # Handle both dict and legacy flat config format
    if isinstance(scheduled_tasks, dict) and "enabled" in scheduled_tasks and "cron" in scheduled_tasks:
        # Legacy format: single task stored at top level
        if scheduled_tasks.get("enabled", False):
            cron_expr = scheduled_tasks.get("cron", "0 9 * * *")
            try:
                scheduler.add_job(
                    run_scheduled_task,
                    "cron",
                    id="email_check",
                    replace_existing=True,
                    args=["email_check", scheduled_tasks],
                    **parse_cron(cron_expr)
                )
                logging.info(f"Scheduled task setup with cron: {cron_expr}")
            except Exception as e:
                logging.error(f"Failed to setup scheduled task: {e}")
    else:
        # New format: multiple tasks
        for task_name, task_config in scheduled_tasks.items():
            if not isinstance(task_config, dict):
                continue
                
            if not task_config.get("enabled", False):
                logging.debug(f"Scheduled task '{task_name}' is disabled")
                continue
            
            cron_expr = task_config.get("cron", "0 9 * * *")
            try:
                scheduler.add_job(
                    run_scheduled_task,
                    "cron",
                    id=f"scheduled_task_{task_name}",
                    replace_existing=True,
                    args=[task_name, task_config],
                    **parse_cron(cron_expr)
                )
                logging.info(f"Scheduled task '{task_name}' setup with cron: {cron_expr}")
            except Exception as e:
                logging.error(f"Failed to setup scheduled task '{task_name}': {e}")


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
