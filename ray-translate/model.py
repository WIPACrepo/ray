import torch
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class TranslationItem(BaseModel):
    fr: str
    to: str
    content: str

app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class Translator:
    def __init__(self, auth_token, max_tokens model_id):
        self.auth_token = auth_token
        self.max_tokens = max_tokens
        self.model_id = model_id
        model_id = "CohereLabs/aya-23-8B"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map="auto",
            token=self.auth_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=self.auth_token
        )

        self.model.to(self.device)

    @app.post("/")
    def translate(self, translation: TranslationItem) -> str:
        # Format message with the command-r-plus chat template
        translate = f"Translate from {translation.fr} to {translation.to}:"
        messages = [{"role": "user", "content": f"{translate} {translation.content}"}]

        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)

        gen_tokens = self.model.generate(
            **input_ids,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=0.3,
            )

        gen_text = self.tokenizer.decode(gen_tokens[0],skip_special_tokens=True)

        return gen_text

parser = argparse.ArgumentParser()

parser.add_argument("--model_id",
                    type=str,
                    help="model id from https://huggingface.co/models")
parser.add_argument("--auth_token",
                    type=str,
                    help="auth token for HuggingFace models")
parser.add_argument("--max_tokens",
                    type=int,
                    default=1,
                    help="max amount of tokens to generatore per request")
translator_app = Translator.bind()

