# utils.py

"""
Basic utils unrelated to clustering

Available function:
- setup_logging: Setup logging configuration
"""

import json
import logging.config
import os

def setup_logging(default_level=logging.INFO,
                  env_key='LOG_CFG',
                  save_path=None,
                  name=None):
    """
    Setup logging configuration

    Params
    ------
    default_level: default logging level
    save_path: filepath to save to (e.g. tasks/image-sent/logs/train.log)
        - If save_path is None, then it assumes there is a logs folder in the same directory as the file being executed
    """
    __cwd__ = os.path.realpath(os.getcwd())
    config_path = os.path.join(__cwd__, 'logging.json')

    path = config_path
    value = os.getenv(env_key, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)

        if os.path.isdir(save_path):
            config['handlers']['info_file_handler']['filename'] = os.path.join(save_path, 'info.log')
            config['handlers']['error_file_handler']['filename'] = os.path.join(save_path, 'error.log')
        else:
            config['handlers']['info_file_handler']['filename'] = save_path
            config['handlers']['error_file_handler']['filename'] = save_path

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger(name=__name__ if name is None else name)

    return logging, logger
