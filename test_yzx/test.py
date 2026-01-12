import openai

client = openai.OpenAI(
  api_key="sk-3Kechw8XbX31d8YuF921C9E9D495417a866cC812FaFe429e",  # Replace with your AIHubMix generated key
  base_url="https://aihubmix.com/v1"
)
prompt='''You are an AI assistant that helps people find information.'''
response = client.chat.completions.create(
  model="qwen2.5-72b-instruct",
  messages=[
      {"role": "user", "content": prompt}
  ],
  temperature=0.2,
)

print(response.choices[0].message.content)