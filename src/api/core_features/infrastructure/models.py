import torch

from injector import singleton, provider, inject, Module
from transformers import AutoModel

from ...domain_features.preprocessing.preprocessing_service import PreprocessingService
from ...domain_features.speech_denoise.denoise_service import DenoiseService
from ...domain_features.transcribe.transcribe_service import TranscribeService
from ...domain_features.transcribe.model_loaders import load_wav2vec2_250_model


class ModelModule(Module):
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @inject
    @provider
    @singleton
    def create_denoise_serivce(self, preprocessor: PreprocessingService) -> DenoiseService:
        model = AutoModel.from_pretrained(
            "mispeech/dasheng-denoiser",
            trust_remote_code=True
        ).eval().to(self._DEVICE)

        def denoise(wav):
            return model(wav.to(self._DEVICE))

        return DenoiseService(denoise, preprocessor)

    @provider
    @singleton
    def create_transcribe_service(self) -> TranscribeService:
        transcriber = load_wav2vec2_250_model(self._DEVICE)
        return TranscribeService(transcriber)
