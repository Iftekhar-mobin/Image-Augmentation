from constants import IMAGE_PATH, \
    LABEL_PATH, \
    IMAGE_SAVE_DIR, \
    SPACING_RANGE, \
    GEN_DATASET_NAME, \
    AUGMENTED_KERAS_GEN_DATA, \
    KERAS_SAVE_DIR, \
    IMAGE_HEIGHT, \
    IMAGE_WIDTH, \
    KERAS_AUG_HOR_IMAGES
from methods import load_data, \
    label_mapping, \
    generate_image, \
    save_image, \
    random_seq_generator, \
    augmented_data_generator, \
    plot_samples, \
    load_generated_data, \
    normalize_data


class AugmentedSeqGen:
    def __init__(self):
        # load keras generated augmented dataset
        aug_keras_dataset = load_generated_data(KERAS_SAVE_DIR, AUGMENTED_KERAS_GEN_DATA)
        print('[INFO] Data is loaded from saved directory:', KERAS_SAVE_DIR)
        self.images = aug_keras_dataset['arr_0'].reshape(aug_keras_dataset['arr_0'].shape[0], IMAGE_HEIGHT, IMAGE_WIDTH)
        self.labels = aug_keras_dataset['arr_1']

    def visualize_generated_aug_dataset(self):
        print('[INFO] Data is loaded from saved directory:', KERAS_SAVE_DIR)
        plot_samples(self.images, self.labels)

    # Generate Horizontal image with optimal spacing
    def generate_random_sequence_with_keras_data(self, see_image, num_samples, seq_len, space_range=SPACING_RANGE, image_width=100):
        print('[INFO] Horizontal Image is generating from keras augmented data...')
        # for normalization
        normalized_data = normalize_data(self.images, see_image=see_image)

        random_seq_generator(num_samples, seq_len, normalized_data, label_mapping(self.labels),
                             space_range, image_width, KERAS_AUG_HOR_IMAGES, GEN_DATASET_NAME)


class SeqGen:
    def __init__(self):
        # Load data API
        self.dataset_images, self.labels = load_data(IMAGE_PATH, LABEL_PATH)
        # Generate mapping Image => class Labels
        self.label_maps = label_mapping(self.labels)

    def horizontal_seq_gen(self, input_sequence, space_range=SPACING_RANGE, image_width=100):
        # Generate Horizontal image with optimal spacing
        horizontal_image_array = generate_image(self.dataset_images, self.label_maps,
                                                input_sequence, space_range, image_width)
        save_image(horizontal_image_array, input_sequence, IMAGE_SAVE_DIR)

    def generate_random_sequence(self, num_samples, seq_len, space_range=SPACING_RANGE, image_width=100):
        random_seq_generator(num_samples, seq_len, self.dataset_images, self.label_maps,
                             space_range, image_width, IMAGE_SAVE_DIR, GEN_DATASET_NAME)

    def augment_dataset(self,
                        num_of_data_to_save=10000,
                        zca_whitening=False,
                        rotation_range=10,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        ):
        augmented_data_generator(
            num_of_data_to_save,
            self.dataset_images,
            self.labels,
            KERAS_SAVE_DIR,
            AUGMENTED_KERAS_GEN_DATA,
            f_center=False,
            s_center=False,
            f_std_normalization=False,
            s_std_normalization=False,
            zca_whitening=zca_whitening,
            zca_epsilon=1e-6,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=None,
            shear_range=shear_range,
            zoom_range=0,
            channel_shift_range=0.,
            fill_mode='nearest',
            c_val=0.,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=0.0001,
            preprocessing_function=None,
            validation_split=0.0,
        )
