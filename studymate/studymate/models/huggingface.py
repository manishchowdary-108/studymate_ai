from __future__ import annotations
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

hf_token = os.getenv("HF_API_TOKEN")


def _get_hf_model_id() -> str:
    # default chat model (Mixtral Instruct)
    return os.getenv("HF_TEXT_GEN_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")


def get_hf_client() -> InferenceClient | None:
    if not hf_token:
        return None
    return InferenceClient(model=_get_hf_model_id(), token=hf_token)


def generate_answer(prompt: str) -> str:
    client = get_hf_client()
    if client is None:
        return "[LLM unavailable] Please configure Hugging Face credentials."

    try:
        # For chat / conversational models
        response = client.chat.completions.create(
            model=_get_hf_model_id(),
            messages=[
                {"role": "system", "content": "You are a helpful study assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("HF_TEMPERATURE", "0.2")),
            top_p=float(os.getenv("HF_TOP_P", "0.9")),
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"[Error] Hugging Face API call failed: {e}"
