from flask_restful import Resource

from ...logger import get_logger

logger = get_logger(__name__)


class HealthResource(Resource):
    def get(self):
        return {"status": "OK"}
