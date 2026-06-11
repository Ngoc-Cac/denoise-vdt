import torch

from torchcodec.decoders import AudioDecoder
from injector import inject

from ...logger import get_logger

logger = get_logger(__name__)


class DenoiseService:
    def __init__(self, model):
        self._model = model
        logger.debug("Initialised template service")

    def denoise_audio(self, file):
        decoder = AudioDecoder(file)

        samples = decoder.get_all_samples()
        sr = samples.sample_rate

        with torch.no_grad():
            denoised = self._model(samples.data)

        return denoised, sr
