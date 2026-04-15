import torch
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
import pathlib
from dataclasses import dataclass
from wipac_dev_tools import from_environment_as_dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class TranslationItem(BaseModel):
    fr: str
    to: str
    content: str

@dataclass
class EnvConfig:
    MODEL_ID: str = None
    MODEL_PATH: pathlib.Path = None
    AUTH_TOKEN: str = None
    MAX_GENERATE_TOKENS: int = 1000

    def __post_init__(self) -> None:
        if self.MODEL_ID is not None and self.MODEL_PATH is not None:
            raise ValueError("Exclusive parameters 'MODEL_ID' and 'MODEL_PATH' defined")
        if self.MODEL_PATH:
            if not pathlib.Path.exists(self.MODEL_PATH):
                raise ValueError(f"Path {self.MODEL_PATH} does not exist")

ENV = from_environment_as_dataclass(EnvConfig)

app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class Translator:
    def __init__(self):
        self.auth_token = ENV.AUTH_TOKEN
        self.max_tokens = ENV.MAX_GENERATE_TOKENS
        self.model_id = ENV.MODEL_PATH if ENV.MODEL_PATH != None else ENV.MODEL_ID
        self.is_local = True if ENV.MODEL_PATH != None else False

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map="auto",
            token=self.auth_token,
            local_files_only = self.is_local
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.auth_token,
            local_files_only = self.is_local
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

translator_app = Translator.bind()

