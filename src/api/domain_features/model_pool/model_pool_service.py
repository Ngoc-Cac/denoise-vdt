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
        self._denoiser = AutoModel.from_pretrained(
            "mispeech/dasheng-denoiser",
            trust_remote_code=True
        ).eval().to(self._DEVICE)
        self._loaded_ckpt = 'baseline'

        self._DENOISERS_STATE_DICTS['baseline'] = self._denoiser.second_encoder.state_dict()
        self._DENOISERS_STATE_DICTS["checkpoint-9963"] = safetensors.torch.load_file(
            config['CKPT_DIR'] / "checkpoint-9963.safetensors"
        )
        self._DENOISERS_STATE_DICTS["checkpoint-15621"] = safetensors.torch.load_file(
            config['CKPT_DIR'] / "checkpoint-15621.safetensors"
        )

        cf_model = self._STT_MODELS["ChunkFormer"]
        cf_model[0] = cf_model[2](self._DEVICE)

        logger.debug("Initialised model pool")

    def load_denoising_ckpt(self, ckpt_name):
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

    def get_transcriber(self, names: list[str]):
        models = []
        for name in names:
            if name in self._STT_MODELS and self._STT_MODELS[name][0] is None:
                model = self._STT_MODELS[name]
                model[0] = model[2](self._DEVICE)
            
            models.append(self._STT_MODELS.get(name, [None])[:2])
        return models
