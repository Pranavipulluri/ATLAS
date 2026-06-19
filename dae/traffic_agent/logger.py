"""
Centralized logging for all components.
Easy to enable/disable per component.
"""
import logging
import sys
import os

# Get log level from environment
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logger(name: str, level=None):
    """
    Create a logger for a module.

    Args:
        name: Module name (e.g., "node", "mqtt", "api")
        level: Log level (default: INFO from env)

    Returns:
        logger instance
    """
    if level is None:
        level = getattr(logging, LOG_LEVEL, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add handler if not already present (avoid duplicates)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Module loggers
logger_node = setup_logger("node")
logger_network = setup_logger("network")
logger_mqtt = setup_logger("mqtt")
logger_api = setup_logger("api")
logger_detection = setup_logger("detection")
logger_sim = setup_logger("sim")
