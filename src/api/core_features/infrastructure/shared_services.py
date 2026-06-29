from injector import singleton, provider, Module

from ...config import Config
from ...domain_features.model_pool.model_pool_service import ModelPoolService
from ...domain_features.preprocessing.preprocessing_service import PreprocessingService


class SharedServiceModule(Module):
    @provider
    @singleton
    def create_model_pool(self, config: Config) -> ModelPoolService:
        return ModelPoolService(config)

    @provider
    @singleton
    def create_preprocessing_service(self) -> PreprocessingService:
        return PreprocessingService()
