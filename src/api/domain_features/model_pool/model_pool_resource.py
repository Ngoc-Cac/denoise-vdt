from flask_restful import Resource, reqparse
from injector import inject

from .model_pool_service import ModelPoolService
from ...logger import get_logger

logger = get_logger(__name__)


class ModelPoolResource(Resource):
    """
    Resource for hosting the model pool service.

    Available methods:

    - `GET`: Get the current state of the model pool. Should return:
        - `loaded_denoiser`: Name of the loaded denoising model.
        - `denoising_models`: Names of supported denoising model.
        - `stt_models`: Names of supported speech-to-text model.

    - `POST`: Load a checkpoint for the denoising model:
        - `ckpt_name`: Name of the checkpoint to load.
    """
    @inject
    def __init__(self, model_pool_service: ModelPoolService):
        self.service = model_pool_service

        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            'ckpt_name',
            type=str,
            location='form',
            required=True,
            help='Checkpoint is required to load denoising model'
        )

    def get(self):
        return self.service.get_supported_models(), 200

    def post(self):
        args = self.reqparse.parse_args()
        ckpt_name = args['ckpt_name']

        loaded = self.service.load_denoising_ckpt(ckpt_name)
        return (
            {"message": f"Loaded checkpoint {ckpt_name} successfully!"},
            200
        ) if loaded else (
            {"message": f"Checkpoint {ckpt_name} could not be loaded"},
            422
        )
