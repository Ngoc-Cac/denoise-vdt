import torch

from ..preprocessing.preprocessing_service import PreprocessingService
from ...logger import get_logger

logger = get_logger(__name__)


class DenoiseService:
    def __init__(self, model, preprocessor: PreprocessingService):
        self._model = model
        self._preprocessor = preprocessor
        logger.debug("Initialised denoising service")

    @torch.no_grad()
    def denoise_audio(self, file):
        audio_chunks = [
            self._model(audio_chunk).cpu()
            for audio_chunk, _ in self._preprocessor.load_file(file)
        ]
        return torch.concat(audio_chunks, dim=-1), 16000
