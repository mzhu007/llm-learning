from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

response = client.files.create(
    file=open(batch_file, "rb"),
    purpose="batch"
)

batch = client.batches.create(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=response.id,
)