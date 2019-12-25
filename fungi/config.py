import os
import yaml
from fungi.validators import get_validator


def get_config_file():
    '''

    '''
    CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.yml')

    if not os.path.isfile(CONFIG_FILE):
        raise OSError('Config file is missing.')

    return CONFIG_DIR, CONFIG_FILE


def validate_config(cfg):
    ''' Validates that the config passes validators
    Input
    ---------------
    cfg: dict
        Nested dictionary usually coming from load_default_config()

    Will Raise
    ---------------
    ValueError when config is not valid.
    '''
    VALIDATORS = get_validator()
    for section, validator in VALIDATORS.items():
        valid = validator.validate(cfg.get(section))
        if not valid:
            print('{} not valid'.format(section))


def load_default_config():
    _, config_FILE = get_config_file()

    with open(config_FILE) as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

    validate_config(cfg)

    return cfg
