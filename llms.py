from langchain_aws import ChatBedrock

REGION = "us-west-2"

ANTHROPIC_MODELS = {
    "Claude 3.7 Sonnet": "arn:aws:bedrock:us-west-2::inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "Claude 3.5 Sonnetv2": "arn:aws:bedrock:us-west-2::inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
}

llm = ChatBedrock(
    region_name=REGION,
    model_id=ANTHROPIC_MODELS["Claude 3.7 Sonnet"],
    provider="anthropic"
)
