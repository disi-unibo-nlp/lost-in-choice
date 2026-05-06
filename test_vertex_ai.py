from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project='project-417aacc6-490b-43c9-9da',
    location='global',
    http_options=types.HttpOptions(api_version='v1')
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="How does AI work?",
)
print(response.text)