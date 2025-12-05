import os
from openai import OpenAI

def get_lmstudio_client():
    return OpenAI(
        base_url=os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1"),
        api_key="not-needed"
    )
