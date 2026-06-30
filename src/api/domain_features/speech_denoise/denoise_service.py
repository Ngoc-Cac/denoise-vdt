import time
import torch

from injector import inject

from ..preprocessing.preprocessing_service import PreprocessingService
from ..model_pool.model_pool_service import ModelPoolService
from ...logger import get_logger

logger = get_logger(__name__)


class DenoiseService:
    @inject
    def __init__(
        self,
        model_pool: ModelPoolService,
        preprocessor: PreprocessingService,
    ):
        self._model_pool = model_pool
        self._preprocessor = preprocessor
        logger.debug("Initialised denoising service")

    @torch.no_grad()
    def denoise_audio(self, file: str):
        denoiser = self._model_pool.get_denoiser()

        logger.info(f"Denoising file {file}")

        start = time.perf_counter_ns()
        audio_chunks = [
            denoiser(audio_chunk).cpu()
            for audio_chunk, _ in self._preprocessor.load_file(file)
        ]
        end = time.perf_counter_ns() - start

        logger.info(f"Denoised file {file} within {end / 1e6:.3f} ms")
        return torch.concat(audio_chunks, dim=-1), 16000
