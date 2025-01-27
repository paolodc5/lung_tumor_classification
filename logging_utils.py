import logging
from logging.handlers import RotatingFileHandler
from colorlog import ColoredFormatter


class Logger:
    def __init__(self, name: str = "app_logger", log_file: str = "app.log", level: int = logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.hasHandlers():
            # File handler
            #file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
            file_handler = logging.FileHandler(log_file,mode="w")
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = ColoredFormatter(
                fmt="%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "white",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
            console_handler.setFormatter(console_formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            self.logger.info("Logger initialized correctly\n")

    def get_logger(self) -> logging.Logger:
        return self.logger


def configure_keras_logging(app_logger: logging.Logger):
    """Configura il logger di Keras per inoltrare i log a `app_logger`."""
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.handlers = []

    class KerasLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            if record.levelno >= logging.ERROR:
                app_logger.error(log_entry)
            elif record.levelno >= logging.WARNING:
                app_logger.warning(log_entry)
            elif record.levelno >= logging.INFO:
                app_logger.info(log_entry)
            else:
                app_logger.debug(log_entry)

    keras_handler = KerasLogHandler()
    keras_handler.setLevel(logging.DEBUG)
    tf_logger.addHandler(keras_handler)
    tf_logger.setLevel(logging.INFO)




# Inizializza il logger globale
app_logger = Logger().get_logger()

