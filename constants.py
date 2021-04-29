import os
# Define the constant parameters
DATASET_DIR = os.path.join(os.getcwd(), 'data')
IMAGE_FILE = 't10k-images.idx3-ubyte'
LABEL_FILE = 't10k-labels.idx1-ubyte'
IMAGE_PATH = os.path.join(DATASET_DIR, IMAGE_FILE)
LABEL_PATH = os.path.join(DATASET_DIR, LABEL_FILE)
IMAGE_SAVE_DIR = os.path.join(os.getcwd(), 'aug_gen_data')
SPACING_RANGE = {'min': 0, 'max': 10}
GEN_DATASET_NAME = 'augmented_data'
AUGMENTED_KERAS_GEN_DATA = 'keras_augmented_data'
KERAS_SAVE_DIR = os.path.join(os.getcwd(), 'keras_data')
KERAS_AUG_HOR_IMAGES = os.path.join(os.getcwd(), 'keras_hor_images')
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28


