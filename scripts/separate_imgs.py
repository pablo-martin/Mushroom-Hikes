import os
import shutil
import configparser
config = configparser.ConfigParser()

config.read('~/scripts/defaults.config')
print(config.sections())


MIN_IMAGES_PER_CLASS = int(config['files']['MIN_IMAGES_PER_CLASS'])
BASE_DIR = str(config['files']['BASE_DIR'])
DATA_DIR = os.path.sep.join([BASE_DIR, str(config['files']['DATA_DIR'])])
DISCARDED_DATA_DIR = os.path.sep.join([BASE_DIR,
                                str(config['files']['DISCARDED_DATA_DIR'])])


main_data = {k: len(os.listdir(os.path.sep.join([DATA_DIR, k]))) \
                                                for k in os.listdir(DATA_DIR)}
side_data = {k: len(os.listdir(os.path.sep.join([DISCARDED_DATA_DIR, k]))) \
                                        for k in os.listdir(DISCARDED_DATA_DIR)}

main_data_to_discard = \
                [w[0] for w in main_data.items() if w[1] < MIN_IMAGES_PER_CLASS]
side_data_to_promote = \
                [w[0] for w in side_data.items() if w[1] > MIN_IMAGES_PER_CLASS]

#moving folders with not enough data to discarded dir
for k in main_data_to_discard:
    shutil.move(os.path.sep.join([DATA_DIR, k]),
                                    os.path.sep.join([DISCARDED_DATA_DIR, k]))

#moving folder with enough data from discarded dir to main
for k in side_data_to_promote:
    shutil.move(os.path.sep.join([DISCARDED_DATA_DIR, k]),
                                    os.path.sep.join([DATA_DIR, k]))
