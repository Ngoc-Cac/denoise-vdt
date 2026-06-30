from src.api.app import create_app
from src.api.logger import get_logger

logger = get_logger(__name__)

app = create_app()

logger.info("Starting application")
app.run(
    host="0.0.0.0",
    port="8080",
    debug=True,
    threaded=True,
    exclude_patterns=['*web_ui*', '*gradio-app.py*']
)
