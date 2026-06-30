import torch
import safetensors.torch

from transformers import AutoModel
from injector import inject

from .stt_model_loaders import (
    load_wav2vec2_250_model,
    load_chunkformer,
    load_PhoWhisper_model
)
from ...config import Config

from typing import Callable
from ...logger import get_logger

logger = get_logger(__name__)


class ModelPoolService:
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DENOISERS_STATE_DICTS = {}
    # transcriber, need_file, loader
    _STT_MODELS = {
        "Wav2Vec2-250h": [None, False, load_wav2vec2_250_model],
        "ChunkFormer": [None, True, load_chunkformer]
    }

    @inject
    def __init__(self, config: Config):

        logger.info("Loading denoising backbone...")
        self._denoiser = AutoModel.from_pretrained(
            "mispeech/dasheng-denoiser",
            trust_remote_code=True
        ).eval().to(self._DEVICE)
        self._loaded_ckpt = 'baseline'

        logger.info("Loading denoising checkpoints...")
        self._DENOISERS_STATE_DICTS['baseline'] = self._denoiser.second_encoder.state_dict()
        self._DENOISERS_STATE_DICTS["checkpoint-9963"] = safetensors.torch.load_file(
            config['CKPT_DIR'] / "checkpoint-9963.safetensors"
        )
        self._DENOISERS_STATE_DICTS["checkpoint-15621"] = safetensors.torch.load_file(
            config['CKPT_DIR'] / "checkpoint-15621.safetensors"
        )

        logger.info(f"Loading default speech-to-text {config['DEFAULT_STT']}...")
        cf_model = self._STT_MODELS[config['DEFAULT_STT']]
        cf_model[0] = cf_model[2](self._DEVICE)

        logger.debug("Initialised model pool")

    def load_denoising_ckpt(self, ckpt_name: str) -> bool:
        """
        Load denoising checkpoint.

        :param str ckpt_name: The name of the checkpoint to load. This should
            be a name from `ModelPoolService.get_supported_models()`.
        :return: Whether the checkpoint was loaded.
        :rtype: bool
        """
        ckpt_exists = ckpt_name in self._DENOISERS_STATE_DICTS
        if ckpt_exists and self._loaded_ckpt != ckpt_name:
            self._loaded_ckpt = ckpt_name
            self._denoiser.second_encoder.load_state_dict(
                self._DENOISERS_STATE_DICTS[ckpt_name]
            )
        return ckpt_exists

    def _denoise(self, wf):
        return self._denoiser(wf.to(self._DEVICE))

    def get_supported_models(self):
        return {
            "loaded_denoiser": self._loaded_ckpt,
            "denoising_models": list(self._DENOISERS_STATE_DICTS.keys()),
            "stt_models": list(self._STT_MODELS.keys())
        }

    def get_denoiser(self):
        return self._denoise

    def get_transcriber(self, names: list[str]) -> list[list[bool, Callable]]:
        """
        Get STT model specified with `names`.

        :param list[str] names: The names of the STT models to retrieve. This
            should be one of the name given by
            `ModelPoolService.get_supported_models()`.
        :return: The retrieved models. Each row in this list is a tuple
            `(need_file, transcriber)`. `transcriber` is the callable for
            transcribing, whereas `need_file` indicates whether the trascriber
            receive a file or waveform as inputs.
        :rtype: list[list[bool, Callable]]
        """
        models = []
        for name in names:
            if name in self._STT_MODELS and self._STT_MODELS[name][0] is None:
                logger.info(f"Loading speech-to-text model {name}")
                model = self._STT_MODELS[name]
                model[0] = model[2](self._DEVICE)
            
            models.append(self._STT_MODELS.get(name, [False, None])[:2])
        return models
