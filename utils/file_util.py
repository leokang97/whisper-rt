import logging
import os
from datetime import datetime

logger = logging.getLogger('file_util')


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        logger.warning("Error: Failed to create the directory.")


def make_filename(prefix, ext):
    dt = datetime.now().strftime('%y%m%d')
    return '%s_%s.%s' % (prefix, dt, ext)
