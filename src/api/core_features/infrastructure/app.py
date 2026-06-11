from injector import singleton, provider, Module

from src.api.config import Config, STATIC_CONFIG


class AppModule(Module):
    @provider
    @singleton
    def provide_config(self) -> Config:
        return Config(STATIC_CONFIG)
