from main import SeqGen, AugmentedSeqGen

# initialization for data+library loading and mapping
driver = SeqGen()

# To generate only one sequence
driver.horizontal_seq_gen(
    input_sequence=[2, 4, 1, 7],
    space_range={'min': 0, 'max': 5},
    image_width=200
)

# To generate bulk after augmenting source dataset
driver.generate_random_sequence(
    num_samples=10,
    seq_len=10,
    space_range={'min': 0, 'max': 10},
    image_width=200
)

# Data augmentation and save augmented data
driver.augment_dataset(
    num_of_data_to_save=10000,
    zca_whitening=False,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
)

# visualize generated augmented dataset to see label mapping is ok
augmented = AugmentedSeqGen()
augmented.visualize_generated_aug_dataset()

########################################################################
#    This Step is generate horizontal Images after augmentation        #
########################################################################
# To generate bulk without augmenting source dataset
augmented.generate_random_sequence_with_keras_data(
    see_image=True,
    num_samples=10,
    seq_len=10,
    space_range={'min': 0, 'max': 10},
    image_width=200
)
