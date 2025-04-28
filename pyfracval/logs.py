import logging
import sys
from typing import Literal

TRACE_LEVEL_NUM = 5
TRACE_LEVEL_NAME = "TRACE"
logging.addLevelName(TRACE_LEVEL_NUM, TRACE_LEVEL_NAME)


# Add the trace method to the Logger class
def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(TRACE_LEVEL_NUM, message, args, **kws)


logging.TRACE = TRACE_LEVEL_NUM  # type: ignore
logging.Logger.trace = trace  # type: ignore


def create_logger(log_level: int, log_file: str | None = None) -> logging.Logger:
    """Set up the main logger for the pyfracval package.

    Configures the 'pyfracval' logger with specified level and handlers.
    Removes existing handlers to prevent duplication. Adds colored console
    output and optional file output.

    Parameters
    ----------
    log_level : int
        The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG,
        TRACE_LEVEL_NUM).
    log_file : str | None, optional
        If provided, the path to a file where logs will also be written,
        by default None (log only to console).

    Returns
    -------
    logging.Logger
        The configured 'pyfracval' logger instance.
    """

    log_format = "%(asctime)s - %(levelname)-8s - %(name)-30s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # --- Configure the 'pyfracval' logger directly ---
    logger = logging.getLogger("pyfracval")
    logger.setLevel(log_level)  # Set the level determined by CLI verbosity

    # Remove existing handlers *from this specific logger* to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.propagate = False  # Prevent messages going to root logger (important!)

    # --- Create and add handlers directly to the 'pyfracval' logger ---
    handlers: list[logging.Handler] = []

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)  # Handler also respects the level
    console_handler.setFormatter(
        CustomLogFormatter(fmt=log_format, datefmt=date_format)
    )
    logger.addHandler(console_handler)
    handlers.append(console_handler)  # Keep track if needed

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        logger.addHandler(file_handler)
        print(f"Logging output also to file: {log_file}")

    # logging.basicConfig(
    #     level=log_level,
    #     format=log_format,
    #     datefmt=date_format,
    #     handlers=handlers,
    # )

    # logger = logging.getLogger("pyfracval")
    # logger.propagate = False
    logger.info(f"Logging configured at level: {logging.getLevelName(log_level)}")
    return logger


class CustomLogFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI color codes based on level.

    Inherits from logging.Formatter and overrides the format method
    to prepend level-specific color codes to the log message.

    Attributes
    ----------
    LEVEL_COLORS : dict[int, str]
        Mapping from logging level numbers to ANSI color escape codes.
    reset : str
        ANSI escape code to reset text formatting.
    """

    # ANSI escape codes for colors
    magenta = "\x1b[35m"  # Magenta for TRACE
    green = "\x1b[32m"  # Blue for DEBUG
    grey = "\x1b[38;20m"  # Grey - often needs terminal support for 256 colors
    # Alternative simpler grey: "\x1b[90m"
    blue = "\x1b[34m"  # Blue for INFO
    yellow = "\x1b[33;20m"  # Yellow for WARNING
    red = "\x1b[31;20m"  # Red for ERROR
    bold_red = "\x1b[31;1m"  # Bold Red for CRITICAL
    reset = "\x1b[0m"  # Reset color

    # Define format string - base_format is now passed during init
    # We just define the colors here
    LEVEL_COLORS = {
        TRACE_LEVEL_NUM: magenta,
        logging.DEBUG: green,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def __init__(
        self,
        fmt="%(levelname)s: %(message)s",
        datefmt=None,
        style: Literal["%", "{", "$"] = "%",
    ):
        """Initialize the formatter.

        Parameters
        ----------
        fmt : str, optional
            The base log format string, by default "%(levelname)s: %(message)s".
        datefmt : str | None, optional
            The date format string, by default None.
        style : Literal["%", "{", "$"], optional
            The formatting style, by default "%".
        """
        self._base_fmt = fmt
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self._base_fmt_orig = fmt

    def format(self, record):
        """Format the specified record as text with added color.

        Overrides the base class method to dynamically add color codes
        before standard formatting.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log string with ANSI color codes.
        """

        # Get the color for the record's level
        color = self.LEVEL_COLORS.get(record.levelno, "")
        reset = self.reset if color else ""

        # Store the original format string from the internal style object
        # The _style attribute holds the style object (%Style, {Style, $Style)
        # The _fmt attribute within the style object holds the format string
        orig_fmt = self._style._fmt

        # Temporarily modify the format string for this record
        # Use the stored original base format string for constructing the colored version
        self._style._fmt = f"{color}{self._base_fmt_orig}{reset}"

        # Call the parent Formatter's format method.
        # This will use the modified self._style._fmt correctly.
        formatted_record = super().format(record)

        # Restore the original format string in the style object
        # so it's correct for the next record.
        self._style._fmt = orig_fmt

        return formatted_record
