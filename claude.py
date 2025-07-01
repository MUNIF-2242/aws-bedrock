import boto3
import json

# Create Bedrock Runtime client (specify region)
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Claude 3 model ID (Haiku, Sonnet, or Opus)
model_id = "anthropic.claude-3-haiku-20240307-v1:0"  # change to sonnet/opus if needed

# Define the prompt
prompt = "Write an essay about living on Mars in 100 words."

# Build request in native Claude 3 structure
request_body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.5,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
})

# Invoke Claude 3 with streaming
response = client.invoke_model_with_response_stream(
    modelId=model_id,
    body=request_body
)

# Stream and print output text in real-time
for event in response["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    if chunk.get("type") == "content_block_delta":
        print(chunk["delta"].get("text", ""), end="", flush=True)
