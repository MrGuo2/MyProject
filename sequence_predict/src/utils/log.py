import logging
import logging.config
import os
from tqdm import tqdm


class TqdmHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


logging.TqdmHandler = TqdmHandler
os.makedirs('model', exist_ok=True)
os.makedirs('log', exist_ok=True)
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), 'logging.conf'))
logger = logging.getLogger('common')
