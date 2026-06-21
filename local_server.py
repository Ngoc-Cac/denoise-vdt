from src.api.app import create_app
from src.api.logger import get_logger

logger = get_logger(
    __name__,
    log_filename="api-server.log",
    rotating_file_handler=True
)

app = create_app()

logger.info("Starting application")
app.debug = True
app.run(host="0.0.0.0", port="8080", threaded=True)
