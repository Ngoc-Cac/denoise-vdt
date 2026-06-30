from pathlib import Path

_ROOT = Path(__file__).parents[2]


class Config:
    _DEFAULT_CONFIG = {
        "ROOT_DIR": _ROOT.resolve(),
        "CKPT_DIR": (_ROOT / 'data/dasheng-ckpts/').resolve(),
        "DEFAULT_STT": "ChunkFormer"
    }

    def __init__(self, config: dict | None = None):
        self.config = self._DEFAULT_CONFIG | (config or {})

    def __getitem__(self, item):
        return self.config.get(item, None)
