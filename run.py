import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)
print("Login successful")

from practice import sft_finetune