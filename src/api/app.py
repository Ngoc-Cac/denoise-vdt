import os
import sys

from flask import Flask, jsonify
from flask_cors import CORS
from flask_injector import FlaskInjector
from flask_restful import output_json, Api
from werkzeug.exceptions import default_exceptions, HTTPException

from .core_features.exception.request import ValidationError
from .core_features.infrastructure import AppModule, SharedServiceModule

from .domain_features import (
    HealthResource,
    DenoiseResource,
    TranscribeResource,
    ModelPoolResource
)

import logging
from .logger import get_logger, setup_rotating_handler

logger = get_logger(__name__)


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


def _setup_api(app: Flask):
    api = Api(app, catch_all_404s=True)
    api.representation({"application/json": output_json})

    # API routes
    api.add_resource(HealthResource, "/health", "/api/v1/health")
    api.add_resource(DenoiseResource, "/denoise", "/api/v1/denoise")
    api.add_resource(TranscribeResource, "/transcribe", "/api/v1/transcribe")
    api.add_resource(ModelPoolResource, "/models", "/api/v1/models")

    return api


def create_app(modules=None, log_dir='logs'):
    logger.info("Creating application")

    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = True
    logger.info("Flask App initialised")

    with app.app_context():
        app.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        app.logger.setLevel(logging.DEBUG)
        if log_dir:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            handler = setup_rotating_handler(f"{log_dir}/debug.log")
            app.logger.addHandler(handler)

    app = _setup_json_error_handling(app)

    api = _setup_api(app)

    logger.info("API Configured")

    if not modules:
        modules = [AppModule(), SharedServiceModule()]

    app.injector = FlaskInjector(app=app, modules=modules).injector

    cors = CORS(resources={r"/api/*": {"origins": "*"}})
    cors.init_app(app)

    logger.info("API has fully started!")

    return app
