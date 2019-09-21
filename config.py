train_raw = "/data/train.xlsx"
test_raw = "/data/test.xlsx"
BASE_DIR = 'Hack'

import os
def get_base_dir_by_name(name):
    path = os.getcwd()
    lastchar = path.find(name) + len(name)
    return os.getcwd()[0:lastchar]

base_dir = get_base_dir_by_name(BASE_DIR).replace("\\","/")
