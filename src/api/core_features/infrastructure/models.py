from injector import singleton, provider, Module
from transformers import AutoModel

from ...domain_features.speech_denoise.denoise_service import DenoiseService


class ModelModule(Module):
    @provider
    @singleton
    def provide_config(self) -> DenoiseService:
        model = AutoModel.from_pretrained(
            "mispeech/dasheng-denoiser",
            trust_remote_code=True
        ).eval()
        return DenoiseService(model)
