import os

from .log import logger


def run_command(command):
    """Run a system command, raise RuntimeError if not success.

    Args:
        command: System command.

    Returns:
        None.
    """
    logger.info(f'Run command: {command}.')
    result = os.system(command)
    if result != 0:
        logger.error(f'run command:{command} failed.')
        raise RuntimeError(f'run command failed:"{command}".')
