import logging
import logging.config
import os


def run_command(command):
    """Run a system command, raise RuntimeError if not success.

    Args:
        command: System command.

    Returns:
        None.
    """
    logger = logging.getLogger('common')
    logger.info(f'Run command: {command}.')
    result = os.system(command)
    if result != 0:
        logger.error(f'run command:{command} failed.')
        raise RuntimeError(f'run command failed:"{command}".')
