#!/usr/bin/env python3
"""
Test script to diagnose Open WebUI web search API support.
This helps determine if the web_search parameter is properly supported
through Open WebUI's OpenAI-compatible API endpoint.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import httpx
from openai import AsyncOpenAI


def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ config.yaml not found. Please ensure you're in the gpt-discord-bot directory.")
        sys.exit(1)


async def test_web_search():
    """Test if web search works through the OpenAI API"""
    config = load_config()
    
    # Get Open WebUI provider config
    provider_config = config.get("providers", {}).get("open-webui")
    if not provider_config:
        print("❌ open-webui provider not configured in config.yaml")
        return False
    
    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key", "sk-no-key-required")
    
    print(f"\n🔍 Testing Open WebUI Web Search")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:20]}..." if len(api_key) > 20 else f"API Key: {api_key}")
    
    # Create OpenAI client
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    # Test 1: Regular request without web_search
    print(f"\n[TEST 1] Regular request WITHOUT web_search parameter")
    print("─" * 50)
    try:
        response = await client.chat.completions.create(
            model="gpt-discord-bot",
            messages=[
                {"role": "user", "content": "2024年 OpenAI 有什麼新聞"}
            ],
            max_tokens=100,
            stream=False
        )
        print(f"✓ Request succeeded")
        print(f"  Response: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
    
    # Test 2: Request WITH web_search in extra_body
    print(f"\n[TEST 2] Request WITH web_search=true in extra_body")
    print("─" * 50)
    try:
        response = await client.chat.completions.create(
            model="gpt-discord-bot",
            messages=[
                {"role": "user", "content": "2024年 OpenAI 有什麼新聞"}
            ],
            max_tokens=100,
            stream=False,
            extra_body={"web_search": True}
        )
        print(f"✓ Request succeeded")
        print(f"  Response: {response.choices[0].message.content[:100]}...")
        
        # Check if response mentions it did a search
        response_text = response.choices[0].message.content.lower()
        if any(indicator in response_text for indicator in ["search", "found", "web", "2024", "recent", "latest"]):
            print(f"✓ Response appears to include search results (contains search-related keywords)")
        else:
            print(f"⚠️  Response may NOT include search results (no search-related keywords detected)")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
    
    # Test 3: Direct HTTP request to verify web_search is in the body
    print(f"\n[TEST 3] Direct HTTP POST to inspect request body")
    print("─" * 50)
    try:
        async with httpx.AsyncClient() as http_client:
            payload = {
                "model": "gpt-discord-bot",
                "messages": [
                    {"role": "user", "content": "2024年 OpenAI 有什麼新聞"}
                ],
                "max_tokens": 50,
                "stream": False,
                "web_search": True  # Direct parameter
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}" if api_key != "sk-no-key-required" else {},
                "Content-Type": "application/json"
            }
            headers = {k: v for k, v in headers.items() if v}
            
            print(f"  Request body: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            
            response = await http_client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code == 200:
                print(f"✓ Request succeeded (HTTP {response.status_code})")
                result = response.json()
                print(f"  Response: {result['choices'][0]['message']['content'][:100]}...")
            else:
                print(f"❌ Request failed (HTTP {response.status_code})")
                print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Direct request failed: {e}")
    
    # Recommendations
    print(f"\n📋 DIAGNOSTICS SUMMARY")
    print("─" * 50)
    print("If web search is NOT working despite 'web_search: True' being set:")
    print("1. ✓ Verify Open WebUI Web Search is enabled:")
    print("   - Admin Panel > Settings > Web Search > Enable Web Search")
    print("2. ✓ Verify a search engine is configured:")
    print("   - Admin Panel > Settings > Web Search > Search Engine")
    print("3. ✓ Verify API keys are set for your search engine (SearXNG, Tavily, etc)")
    print("4. ✓ Test in Open WebUI console with a web search query")
    print("5. ✓ Check Open WebUI logs for errors")
    print("\nPossibilities:")
    print("• Open WebUI's OpenAI-compatible API endpoint may NOT support web_search")
    print("• Web search might only work through Open WebUI's native UI")
    print("• The parameter may need to be named differently")
    print("• Web search may require a different request format")
    
    return True


if __name__ == "__main__":
    print("🚀 Open WebUI Web Search Diagnostic Tool")
    print("=" * 50)
    
    try:
        asyncio.run(test_web_search())
        print("\n✓ Diagnostic complete. Check the results above.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
