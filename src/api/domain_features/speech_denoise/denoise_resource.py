import io
import os
import tempfile

from flask import send_file
from flask_restful import Resource, reqparse
from injector import inject
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from torchcodec.encoders import AudioEncoder

from .denoise_service import DenoiseService
from ...logger import get_logger

logger = get_logger(__name__)


class DenoiseResource(Resource):
    """
    Generic tempalte for API Resource.
    """
    @inject
    def __init__(self, template_service: DenoiseService):
        """
        Initialize the resource with dependency-injected service.

        :param TemplateService template_service: The service to use for
            handling requests
        """
        self.denoise_service = template_service

        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            'audio',
            type=FileStorage,
            location='files',
            required=True,
            help='Audio file is required'
        )

    def _audio_to_bytes(self, samples, sample_rate):
        byte_io = io.BytesIO()
        encoder = AudioEncoder(samples.data, sample_rate=sample_rate)
        encoder.to_file_like(byte_io, "wav")
        byte_io.seek(0)  # set to the beginning to stream everything

        return byte_io

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
        """
        Handle POST request.

        :return: A JSON response with HTTP status code
        :rtype: tuple[dict, int]
        """
        args = self.reqparse.parse_args()

        try:
            filepath = self._save_file(args['audio'])

            denoised, sr = self.denoise_service.denoise_audio(filepath)
            return send_file(
                self._audio_to_bytes(denoised, sr),
                "audio/wav",
                download_name="cleaned.wav"
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"success": False, "error": str(e)}, 500
