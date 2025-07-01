import streamlit as st
import boto3
import json

# Streamlit UI
st.title("Claude 3 Streaming - AWS Bedrock")
st.markdown("Enter a prompt below to get a response from Claude 3 (Haiku model)")

prompt = st.text_area("Prompt", "Write an poem about living on Mars in 100 words.")

if st.button("Generate Response"):
    with st.spinner("Generating response from Claude 3..."):

        # AWS Bedrock client
        client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Claude model ID
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        # Build request body
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

        # Prepare placeholder for streaming output
        output_placeholder = st.empty()
        full_response = ""

        # Call Claude with streaming
        try:
            response = client.invoke_model_with_response_stream(
                modelId=model_id,
                body=request_body
            )

            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    delta_text = chunk["delta"].get("text", "")
                    full_response += delta_text
                    output_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error during generation: {e}")
