import os
import tempfile

from flask_restful import Resource, reqparse
from injector import inject
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .transcribe_service import TranscribeService
from ...config import Config

from ...logger import get_logger

logger = get_logger(__name__)


class TranscribeResource(Resource):
    """
    Resource for hosting the transcription endpoint.

    Available methods:

    - `POST`: Upload audio file for transcribing, with arguments:
        - `audio`: File to transcribe. This is required.
        - `models`: The model(s) to use for transcribing. This should be a
        name given by the `ModelPoolService`. See that service for more
        information.
    """
    @inject
    def __init__(self, transcribe_service: TranscribeService, config: Config):
        self.service = transcribe_service

        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            'audio',
            type=FileStorage,
            location='files',
            required=True,
            help='Audio file is required'
        )
        self.reqparse.add_argument(
            'models',
            type=str,
            location='form',
            action='append',
            required=False,
            default=[config["DEFAULT_STT"]]
        )

    def _save_file(self, file):
        tmp_dir = tempfile.gettempdir()
        filename = secure_filename(file.filename or "")
        ext = os.path.splitext(filename)[1]
        fd, filepath = tempfile.mkstemp(
            prefix="upload_", suffix=ext, dir=tmp_dir
        )
        os.close(fd)
        file.save(filepath)

        return filepath

    def post(self):
        args = self.reqparse.parse_args()

        filepath = self._save_file(args['audio'])
        return self.service.transcribe(
            filepath, args['models']
        ), 200
