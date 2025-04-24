import logging


class CustomLogFormatter(logging.Formatter):
    """Custom formatter to add colors based on log level."""

    # ANSI escape codes for colors
    grey = "\x1b[38;20m"  # Grey - often needs terminal support for 256 colors
    # Alternative simpler grey: "\x1b[90m"
    blue = "\x1b[34m"  # Blue for INFO
    yellow = "\x1b[33;20m"  # Yellow for WARNING
    red = "\x1b[31;20m"  # Red for ERROR
    bold_red = "\x1b[31;1m"  # Bold Red for CRITICAL
    reset = "\x1b[0m"  # Reset color

    # Define format string - decide if you want filename/lineno always
    # Basic format:
    base_format = "%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s"
    # Format with file/line number:
    # base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + base_format + reset,
        logging.INFO: blue + base_format + reset,  # Changed INFO to blue
        logging.WARNING: yellow + base_format + reset,
        logging.ERROR: red + base_format + reset,
        logging.CRITICAL: bold_red + base_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(
            record.levelno, self.base_format
        )  # Fallback to base format
        formatter = logging.Formatter(
            log_fmt, datefmt="%Y-%m-%d %H:%M:%S"
        )  # Add datefmt here
        return formatter.format(record)
