# ------------------------------------------------
# Imports
# ------------------------------------------------
import logging
import os
import sys
import random
from shutil import copyfile, copy, copy2


# ------------------------------------------------
# Defines
# ------------------------------------------------
ORIGINAL_DIR = '/Users/thelacker/PycharmProjects/logos/data_new/logos'
TARGET_DIR = 'target'
TRAIN_DIR_NAME = '1_Train'
VALIDATION_DIR_NAME = '2_Validation'
TEST_DIR_NAME = '3_Test'
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.15

# ------------------------------------------------
# Init logging
# ------------------------------------------------
# Levels: debug, info, warn, error, critical
# create logger with 'HR_Scout'
logger = logging.getLogger('Xsplit')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('Xsplit.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# ------------------------------------------------
# copy helper
# ------------------------------------------------
def copy_helper(src_file_path_array, dst_folder_path):
    # create folder if not existent
    try:
        os.stat(dst_folder_path)
    except:
        os.mkdir(dst_folder_path)

        # copy loop
    for src_file_path in src_file_path_array:
        copy2(src_file_path, dst_folder_path)



# ------------------------------------------------
# Import the Logos
# ------------------------------------------------
def import_logos_from_dir(original_dir_path=ORIGINAL_DIR):
    file_path_array = []

    if not original_dir_path:
        logger.critical('no directory')
        sys.exit()
    if not os.path.isdir(original_dir_path):
        logger.critical('no directory ' + original_dir_path + ' not found')
        sys.exit()

    # Folder Loop
    single_dir_num = 0
    single_file_num = 0
    for single_dir_name in os.listdir(original_dir_path):
        if not single_dir_name:
            logger.warning('invalid directory - skipping...')
            continue
        single_dir_path = os.path.join(original_dir_path, single_dir_name)
        if not os.path.isdir(single_dir_path):
            logger.warning('invalid directory - skipping...')
            continue
        print(single_dir_name + '(' + str(single_dir_num) + ')')

        # File Loop
        for single_file_name in os.listdir(single_dir_path):
            if not single_file_name:
                logger.warning('invalid file - skipping...')
                continue
            single_file_path = os.path.join(single_dir_path, single_file_name)
            if not os.path.isfile(single_file_path):
                logger.warning('invalid file - skipping...')
                continue
            print ('- ' + single_file_name)

            # Add path to array
            file_path_array.append(single_file_path)

            single_file_num += 1
        single_dir_num += 1

    print('-------------------')
    #print('Num Files: ' + str(len(file_path_array)))
    #sys.exit()

    # Shuffle
    random.shuffle(file_path_array)

    # Split
    num_vals = len(file_path_array)
    t = int(num_vals * TRAIN_RATIO)
    train_file_path_array       = file_path_array[:t]
    validation_file_path_array  = file_path_array[t:int(num_vals * (TRAIN_RATIO + VALIDATION_RATIO))]
    test_file_path_array        = file_path_array[int(num_vals * (TRAIN_RATIO + VALIDATION_RATIO)):]

    print('ALL: ' + str(len(file_path_array)))
    print('train: ' + str(len(train_file_path_array)))
    print('valid: ' + str(len(validation_file_path_array)))
    print('test: ' + str(len(test_file_path_array)))
    #sys.exit()

    # Copy
    copy_helper(train_file_path_array, os.path.join(TARGET_DIR, TRAIN_DIR_NAME))
    copy_helper(validation_file_path_array, os.path.join(TARGET_DIR, VALIDATION_DIR_NAME))
    copy_helper(test_file_path_array, os.path.join(TARGET_DIR, TEST_DIR_NAME))

    logger.info('finished! number of dirs: ' + str(single_dir_num) + '; number of files: ' + str(single_file_num))

# ------------------------------------------------
# Call
# ------------------------------------------------
import_logos_from_dir()