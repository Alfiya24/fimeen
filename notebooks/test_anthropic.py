from anthropic import Anthropic
from src.config import ANTHROPIC_API_KEY

client = Anthropic(api_key=ANTHROPIC_API_KEY)

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(response.content[0].text)