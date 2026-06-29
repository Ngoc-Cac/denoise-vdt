import os
from pathlib import Path


STATIC_CONFIG = {
    "CKPT_DIR": Path(__file__).resolve().parent.parent.parent / 'data/dasheng-ckpts/'
}


class Config:
    def __init__(self, config: dict):
        self.config = config
        self.is_prod = bool(
            self["IS_PRODUCTION"] and self["IS_PRODUCTION"].lower() == "true"
        )

    def __getitem__(self, item):
        if item in os.environ:
            return os.environ[item]
        return self.config.get(item, None)

    def is_testing(self):
        return not self.is_prod


CONFIG = Config(STATIC_CONFIG)
