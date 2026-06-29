import os
import tempfile

from flask_restful import Resource, reqparse
from injector import inject
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .transcribe_service import TranscribeService
from ...logger import get_logger

logger = get_logger(__name__)


class TranscribeResource(Resource):
    """
    Resource for hosting the transcription endpoint
    """
    @inject
    def __init__(self, template_service: TranscribeService):
        self.service = template_service

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
            default=['ChunkFormer']
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

        try:
            filepath = self._save_file(args['audio'])
            transcripts = self.service.transcribe(
                filepath, args['models']
            )
            return transcripts, 200
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}, 500
