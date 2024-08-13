"""
    This module configures logging for the application.
"""

import logging


def setup_logging():
    """
    This function setups the root logger.
    """

    logger = logging.getLogger("Logger")

    logger.setLevel(logging.INFO)

    # Create a console handler to output logs to the console

    console_handler = logging.StreamHandler()

    # Create a formatter which includes the timestamp

    formatter = logging.Formatter(
        "%(filename)s - %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the formatter for the handler

    console_handler.setFormatter(formatter)

    # Add the handler to the logger

    logger.addHandler(console_handler)

    return logger
