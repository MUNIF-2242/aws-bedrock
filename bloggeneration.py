import streamlit as st
import boto3
import botocore.config
import json
from datetime import datetime

def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt = f"""<s>[INST]Human: Write a 200 words blog on the topic {blogtopic}
Assistant:[/INST]
"""

    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1",
                               config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3}))
        response = bedrock.invoke_model(body=json.dumps(body), modelId="meta.llama3-70b-instruct-v1:0")

        response_content = response.get('body').read()
        response_data = json.loads(response_content)
        blog_details = response_data['generation']
        return blog_details
    except Exception as e:
        st.error(f"Error generating the blog: {e}")
        return ""

def save_blog_details_s3(s3_key, s3_bucket, generated_blog):
    s3 = boto3.client('s3')

    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=generated_blog)
        st.success(f"Blog saved to S3 at {s3_key}")
    except Exception as e:
        st.error(f"Error saving blog to S3: {e}")

# Streamlit UI

st.title("AWS Bedrock Blog Generator")

blog_topic = st.text_input("Enter blog topic")

if st.button("Generate Blog"):
    if blog_topic.strip() == "":
        st.warning("Please enter a blog topic.")
    else:
        with st.spinner("Generating blog..."):
            blog_text = blog_generate_using_bedrock(blog_topic)
            if blog_text:
                st.subheader("Generated Blog")
                st.write(blog_text)

                # Save to S3
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                s3_key = f"blog-output/{current_time}.txt"
                s3_bucket = 'blogbucket.mdmunifhasan.click'  # change to your bucket name

                save_blog_details_s3(s3_key, s3_bucket, blog_text)
            else:
                st.error("Failed to generate blog.")
