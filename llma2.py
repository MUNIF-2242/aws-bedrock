import boto3
import json
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in your AWS Region
client = boto3.client("bedrock-runtime")

# Llama 3 model ID on Bedrock
model_id = "meta.llama3-70b-instruct-v1:0"
""

# The prompt you want to send
prompt = "write an essay for living on mars in 200 words\n\nAssistant:"

# Llama 3 expects this specific instruction formatting
formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Prepare the request body with model-specific parameters
native_request = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5
}

# Convert the request to JSON string
request_body = json.dumps(native_request)

try:
    # Invoke the model
    response = client.invoke_model(modelId=model_id, body=request_body)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# The response body is a stream, so read and decode it
response_body = response["body"].read()

# Parse the JSON response
model_response = json.loads(response_body)

# Extract the generated text from the response
response_text = model_response.get("generation", "")

print(response_text)
