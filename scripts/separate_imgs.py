import os
import shutil
import configparser

def separate_images_from_config(DATA_DIR, DISCARDED_DATA_DIR, MIN_IMAGES_PER_CLASS):
    def ensure_dir_exists(dir_name):
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ensure_dir_exists(DATA_DIR)
    ensure_dir_exists(DISCARDED_DATA_DIR)

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
    return
