import torch

from torchcodec.decoders import AudioDecoder

from ..preprocessing.preprocessing_service import PreprocessingService
from ...logger import get_logger

logger = get_logger(__name__)


class TranscribeService:
    """Service for transcribing speech."""
    def __init__(
        self,
        transcriber,
        preprocessor,
        need_input_file: bool = False,
    ):
        """
        Initialise TranscribeService.

        :param Callable transcribe: A function performing the transcription
            logic. This function should receive two arguments: the waveform and
            its sampling rate. If `need_input_file=True`, a file will be passed
            instead.
        :param Callable need_input_file: Whether or not the transcriber work
            with input files instead of raw samples.
        """
        self._transcriber = transcriber
        self._preprocessor = preprocessor
        self._need_input_file = need_input_file

        logger.debug("Initialised transcribing service")

    @torch.no_grad()
    def transcribe(self, file):
        if self._need_input_file:
            return self._transcriber(file)
        else:
            audio_chunks = [
                self._transcriber(audio_chunk).cpu()
                for audio_chunk, _ in self._preprocessor.load_file(file)
            ]
            return torch.concat(audio_chunks, dim=-1), 16000
