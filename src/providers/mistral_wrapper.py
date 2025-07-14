from src.core.provider import LLMProvider


class MistralWrapper(LLMProvider):
    def __init__(self, infer_fn):
        self.infer = infer_fn

    def __call__(self, prompt: str, **kwargs):
        max_tokens = kwargs.get("max_tokens", 256)
        return self.infer(prompt, max_tokens=max_tokens)