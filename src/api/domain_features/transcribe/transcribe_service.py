import torch

from injector import inject

from ..preprocessing.preprocessing_service import PreprocessingService
from ..model_pool.model_pool_service import ModelPoolService
from ...logger import get_logger

logger = get_logger(__name__)


class TranscribeService:
    """Service for transcribing speech."""
    @inject
    def __init__(
        self,
        model_pool: ModelPoolService,
        preprocessor: PreprocessingService,
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
        self._model_pool = model_pool
        self._preprocessor = preprocessor

        logger.debug("Initialised transcribing service")

    @torch.no_grad()
    def _transcribe(
        self,
        file,
        transcriber,
        need_input_file,
    ):
        if need_input_file:
            return transcriber(file)
        else:
            return ' '.join([
                transcriber(audio_chunk)
                for audio_chunk, _ in self._preprocessor.load_file(file)
            ])

    def transcribe(self, file, model_names):
        return {
            name: self._transcribe(file, *tools)
            for tools, name in zip(
                self._model_pool.get_transcriber(model_names),
                model_names
            )
        }
