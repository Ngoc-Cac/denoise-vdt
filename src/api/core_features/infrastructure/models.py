from injector import singleton, provider, Module
from transformers import AutoModel

from ...domain_features.speech_denoise.denoise_service import DenoiseService


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
