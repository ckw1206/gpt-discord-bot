# Script to fix llmcord.py - remove [Voice message:] prefix
# Run this from Windows: docker exec gpt-discord-bot python /app/bot/test/fix_llmcord.py

import re

with open('/app/llmcord.py', 'r') as f:
    content = f.read()

# Replace the voice message prefix
old = '[Voice message: {transcribed}]'
new = '{transcribed}'
content = content.replace(old, new)

with open('/app/llmcord.py', 'w') as f:
    f.write(content)

print('Done - removed [Voice message:] prefix')