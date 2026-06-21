import torch

from injector import singleton, provider, Module
from torchaudio import functional as F
from transformers import AutoModel

from ...domain_features.speech_denoise.denoise_service import DenoiseService
from ...domain_features.transcribe.transcribe_service import TranscribeService
from ...domain_features.transcribe.model_loaders import load_wav2vec2_250_model


class ModelModule(Module):
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @provider
    @singleton
    def create_denoise_serivce(self) -> DenoiseService:
        model = AutoModel.from_pretrained(
            "mispeech/dasheng-denoiser",
            trust_remote_code=True
        ).eval().to(self._DEVICE)

        def denoise(wav, sr):
            wav = F.resample(wav, sr, 16000)
            return model(wav.to(self._DEVICE)), 16000

        return DenoiseService(denoise)

    @provider
    @singleton
    def create_transcribe_service(self) -> TranscribeService:
        transcriber = load_wav2vec2_250_model(self._DEVICE)
        return TranscribeService(transcriber)
