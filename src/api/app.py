import logging
import logging.config
import sys
from os import environ

import flask_restful
from flask import Flask, jsonify
from flask_cors import CORS
from flask_injector import FlaskInjector
from flask_restful import output_json
from werkzeug.exceptions import default_exceptions, HTTPException

from .core_features.exception.request import ValidationError
from .core_features.infrastructure.modules import create_modules

from .logger import get_logger

from .domain_features.health.health_resource import HealthResource
from .domain_features.speech_denoise.denoise_resource import DenoiseResource


logger = get_logger(__name__)


class Api(flask_restful.Api):
    def __init__(self, *args, **kwargs):
        super(Api, self).__init__(*args, **kwargs)
        self.representations = {
            "application/json": output_json
        }


def _setup_json_error_handling(app: Flask):
    def make_json_error(ex):
        message = str(ex.message) if hasattr(ex, "message") else str(ex)
        app.logger.exception(ex)

        data = {
            "type": ex.__class__.__name__,
            "message": message
        }

        if isinstance(ex, HTTPException):
            data["code"] = ex.code
            res = jsonify(data)
            res.status_code = ex.code
        else:
            res = jsonify(data)
            res.status_code = 400 if isinstance(ex, ValidationError) else 500

        return res

    for code in default_exceptions.keys():
        app.register_error_handler(code, make_json_error)

    app.register_error_handler(Exception, make_json_error)

    return app


def create_app(modules=None):
    logger.info("Creating application")

    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = True
    logger.info("Flask App ready")

    api = Api(app, catch_all_404s=True)

    with app.app_context():
        app.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        app.logger.setLevel(logging.DEBUG)

    app = _setup_json_error_handling(app)

    # API routes
    api.add_resource(HealthResource, "/health", "/api/v1/health")
    api.add_resource(DenoiseResource, "/denoise", "/api/v1/denoise")

    logger.info("API Configured")

    if not modules:
        modules = create_modules()

    app.injector = FlaskInjector(app=app, modules=modules).injector

    cors = CORS(resources={
        r"/api/*": {"origins": "*"},
    })
    cors.init_app(app)

    logger.info("API has fully started!")
    logger.debug("Current env %s", environ)

    return app
