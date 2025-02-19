from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=input("> "),
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                google_search=types.GoogleSearchRetrieval(
                    dynamic_retrieval_config=types.DynamicRetrievalConfig(
                        dynamic_threshold=0.5
                    )
                )
            )
        ]
    ),
)
print(response.text)
