from starlette.requests import Request

from ray import serve

from transformers import pipeline

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Classification:
    def __init__(self):
        # Load model
        self.model = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    def classify(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        scored = model_output[0]["score"]
        labeled = model_output[0]["label"]

        return f"Label: {labeled}, Score: {scored}"

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.classify(english_text)

translator_app = Classification.bind()