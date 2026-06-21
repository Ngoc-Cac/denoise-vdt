import torch

from torchcodec.decoders import AudioDecoder

from ...logger import get_logger

logger = get_logger(__name__)


class TranscribeService:
    """Service for transcribing speech."""
    def __init__(self, transcriber, need_input_file: bool = False):
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
        self._need_input_file = need_input_file

        logger.debug("Initialised template service")

    def transcribe(self, file):
        if self._need_input_file:
            inputs = (file,)
        else:
            decoder = AudioDecoder(file)

            samples = decoder.get_all_samples()
            inputs = samples.data.mean(dim=0), samples.sample_rate

        with torch.no_grad():
            return self._transcriber(*inputs)
