from flask_restful import Resource, reqparse
from injector import inject

from .model_pool_service import ModelPoolService
from ...logger import get_logger

logger = get_logger(__name__)


class ModelPoolResource(Resource):
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

        try:
            loaded = self.service.load_denoising_ckpt(ckpt_name)

            return (
                {"message": f"Loaded checkpoint {ckpt_name} successfully!"},
                200,
            ) if loaded else (
                {"message": f"Checkpoint {ckpt_name} could not be loaded"},
                422
            )
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}, 500
