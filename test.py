import requests, os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    raise ValueError("❌ No HF_API_TOKEN found in .env file")

headers = {"Authorization": f"Bearer {hf_token}"}
url = "https://huggingface.co/api/whoami-v2"

r = requests.get(url, headers=headers)

if r.status_code == 200:
    print("✅ Hugging Face token is valid!")
    print("Response:", r.json())
else:
    print(f"❌ Token test failed. Status {r.status_code}")
    print(r.text)
