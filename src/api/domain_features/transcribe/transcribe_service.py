import torch

from injector import inject

from ..preprocessing.preprocessing_service import PreprocessingService
from ..model_pool.model_pool_service import ModelPoolService

from typing import Callable
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

        :param ModelPoolService model_pool: The shared service that hosts/loads
            model. This service will refer to the `model_pool` to retireve the
            relevant model for transcribing.
        :param PreprocessingService preprocessor: The shared service that
            handles loading audio files efficiently. This service refers to the
            `preprocessor` to load the audio file without impacting performance.
            NOTE: Any model-specific preprocessing should NOT be handled by the
            `preprocessor`.
        """
        self._model_pool = model_pool
        self._preprocessor = preprocessor

        logger.debug("Initialised transcribing service")

    @torch.no_grad()
    def _transcribe(
        self,
        file: str,
        transcriber: Callable,
        need_input_file: bool,
    ) -> str:
        # specify need_input_file for models which works with file directly
        return transcriber(file) if need_input_file else ' '.join([
            transcriber(audio_chunk)
            for audio_chunk, _ in self._preprocessor.load_file(file)
        ])

    def transcribe(self, file: str, model_names: list[str]) -> dict[str, str]:
        """
        Transcribe the audio file `file` using the specified `model_names` models.

        :param str file: The path to the audio file.
        :param list[str] model_names: The names of the Speech-to-text model to
            use. This should be a name that the shared `ModelPoolService`
            recognises. Use `ModelPoolService.get_supported_models()` to inspect
            the models' names.
        :return: The transcription.
        :rtype: str
        """
        logger.info(f"Transcribing file {file}")
        return {
            name: self._transcribe(file, *tools)
            for tools, name in zip(
                self._model_pool.get_transcriber(model_names),
                model_names
            )
        }
