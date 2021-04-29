import os
from mlxtend.data import loadlocal_mnist
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler


# Normalize dataset between 0~1
def normalize_data(dataset, see_image=True):
    if see_image:
        dataset *= (255.0 / dataset.max())
        return dataset
    else:
        # reshaping to fit into MinMaxScaler 3D=>2D=>3D
        sample_size, x, y = dataset.shape
        train_dataset = dataset.reshape((sample_size, x * y))

        scale = MinMaxScaler()
        scale.fit(train_dataset)
        dataset = scale.transform(train_dataset)
        dataset = dataset.reshape(sample_size, x, y)

        return dataset


# http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
# return label and 10000 Images Pixel Map 28x28
def load_data(path_img, path_lbl):
    images, labels = loadlocal_mnist(
        images_path=path_img,
        labels_path=path_lbl
    )
    # convert data to 32 bit floating point
    images = images.reshape(10000, 28, 28).astype('float32')
    return images, labels


# return class map with Image data 28x28
def label_mapping(dataset_labels):
    # Total classes 10, labels 0~9
    class_map = [[] for i in range(10)]
    # For i=0 it will be [[], [], [], [], [], [], [], [0], [], []]
    # dataset_labels[0] = 7
    for i in range(len(dataset_labels)):
        class_map[dataset_labels[i]].append(i)

    return class_map


def _space_generator(length_input_sequence, image_width, image_height, spacing_range):
    number_of_space = length_input_sequence - 1
    number_of_digits = length_input_sequence
    auto_space = (image_width - image_height * number_of_digits) / number_of_space
    if auto_space < 0:
        auto_space = 1
        print('[INFO] Image_width is too short to fit (digits+space), Spacing is selected 1px')
        return auto_space
    elif spacing_range['min'] <= auto_space <= spacing_range['max']:
        return int(auto_space)
    else:
        print('[INFO] ' + str(int(auto_space)) + 'px space is needed to fit given width', image_width,
              ', allowed:', spacing_range)
        auto_space = random.randint(spacing_range['min'], spacing_range['max'])
        print('[INFO] Choosing random space: ', str(auto_space) + 'px')
        return auto_space


def generate_image(dataset_images, label_maps, input_sequence,
                   spacing_range, image_width, image_height=28):
    # find optimal space automatically
    auto_spacing = _space_generator(len(input_sequence), image_width, image_height, spacing_range)
    # generate a array for space (image_height=28, space)
    spacing = np.ones(image_height * auto_spacing, dtype='float32').reshape(image_height, auto_spacing)

    # find the index of the given class - label
    # We are choosing samples index randomly
    label_index = random.choice(label_maps[input_sequence[0]])
    # get the image mapping from dataset [28x28] matrix
    image = dataset_images[label_index]
    # stacking one after another [28x28] + [28x5] = [28x33]
    whole_image = np.hstack((image, spacing))
    # Making an array like this: [28x33] + [28x33] + [28x33] + [28x28]
    sequence_length = len(input_sequence)
    for i in range(1, sequence_length):
        label_index = random.choice(label_maps[input_sequence[i]])
        if i < sequence_length - 1:
            dataset_image = dataset_images[label_index]
            temp_image = np.hstack((dataset_image, spacing))
            whole_image = np.hstack((whole_image, temp_image))
        else:
            dataset_image = dataset_images[label_index]
            whole_image = np.hstack((whole_image, dataset_image))

    return whole_image


def save_image(image_array, input_sequence, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_name = "".join(list(map(str, input_sequence)))
    image = Image.fromarray(image_array)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(os.path.join(save_dir, img_name + ".png"))
    print('[INFO] ' + img_name + '.png is saved in', save_dir)


def save_array(pixels_map, labels, save_dir, dataset_name):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = os.path.join(save_dir, dataset_name)
    np.savez(file_name, pixels_map, labels)
    print('[INFO] Array is saved for future use in:', file_name + '.npz')


def load_generated_data(path, dataset_name):
    dataset = os.path.join(path, dataset_name + '.npz')
    if os.path.exists(dataset):
        # generated_data['arr_0'] generated_data['arr_1']
        return np.load(dataset)
    else:
        print('[INFO] Keras augmented dataset is not generated yet ...')


def random_seq_generator(num_samples, seq_len, dataset_images, label_maps,
                         spacing, image_width, save_dir, to_save_name):
    pixel_map = []
    labels = []
    for i in range(num_samples):
        seq_values = np.random.randint(0, 10, seq_len)
        seq = generate_image(dataset_images, label_maps, seq_values, spacing, image_width)
        pixel_map.append(seq)
        labels.append(seq_values)
        save_image(seq, seq_values, save_dir)

    save_array(pixel_map, labels, save_dir, to_save_name)


def _show_and_save_samples(generator, image_dataset, labels, save_dir):
    print("[INFO] generating sample images...")
    for x_batch, _ in generator.flow(
            image_dataset,
            labels,
            batch_size=9,
            save_to_dir=save_dir,
            save_prefix='aug',
            save_format='png'
    ):

        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.show()
        break


def augmented_data_generator(
        num_of_data_to_save,
        image_dataset,
        labels,
        save_dir,
        saved_file_name,
        f_center=False,
        s_center=False,
        f_std_normalization=False,
        s_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=None,
        shear_range=0.2,
        zoom_range=0,
        channel_shift_range=0.,
        fill_mode='nearest',
        c_val=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=0.0001,
        preprocessing_function=None,
        validation_split=0.0,
):
    # reshape to be [samples]width][height][channels]
    image_dataset = image_dataset.reshape(image_dataset.shape[0], 28, 28, 1)
    '''  Arguments:
      featurewise_center: Boolean.
          Set input mean to 0 over the dataset, feature-wise.
      samplewise_center: Boolean. Set each sample mean to 0.
      featurewise_std_normalization: Boolean.
          Divide inputs by std of the dataset, feature-wise.
      samplewise_std_normalization: Boolean. Divide each input by its std.
      zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
      zca_whitening: Boolean. Apply ZCA whitening.
      rotation_range: Int. Degree range for random rotations.
      width_shift_range: Float, 1-D array-like or int
          - float: fraction of total width, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-width_shift_range, +width_shift_range)`
          - With `width_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `width_shift_range=[-1, 0, +1]`,
              while with `width_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      height_shift_range: Float, 1-D array-like or int
          - float: fraction of total height, if < 1, or pixels if >= 1.
          - 1-D array-like: random elements from the array.
          - int: integer number of pixels from interval
              `(-height_shift_range, +height_shift_range)`
          - With `height_shift_range=2` possible values
              are integers `[-1, 0, +1]`,
              same as with `height_shift_range=[-1, 0, +1]`,
              while with `height_shift_range=1.0` possible values are floats
              in the interval [-1.0, +1.0).
      brightness_range: Tuple or list of two floats. Range for picking
          a brightness shift value from.
      shear_range: Float. Shear Intensity
          (Shear angle in counter-clockwise direction in degrees)
      zoom_range: Float or [lower, upper]. Range for random zoom.
          If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
      channel_shift_range: Float. Range for random channel shifts.
      fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
          Default is 'nearest'.
          Points outside the boundaries of the input are filled
          according to the given mode:
          - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
          - 'nearest':  aaaaaaaa|abcd|dddddddd
          - 'reflect':  abcddcba|abcd|dcbaabcd
          - 'wrap':  abcdabcd|abcd|abcdabcd
      cval: Float or Int.
          Value used for points outside the boundaries
          when `fill_mode = "constant"`.
      horizontal_flip: Boolean. Randomly flip inputs horizontally.
      vertical_flip: Boolean. Randomly flip inputs vertically.
      rescale: rescaling factor. Defaults to None.
          If None or 0, no rescaling is applied,
          otherwise we multiply the data by the value provided
          (after applying all other transformations).
      preprocessing_function: function that will be applied on each input.
          The function will run after the image is resized and augmented.
          The function should take one argument:
          one image (Numpy tensor with rank 3),
          and should output a Numpy tensor with the same shape.
      data_format: Image data format,
          either "channels_first" or "channels_last".
          "channels_last" mode means that the images should have shape
          `(samples, height, width, channels)`,
          "channels_first" mode means that the images should have shape
          `(samples, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      validation_split: Float. Fraction of images reserved for validation
          (strictly between 0 and 1).
      dtype: Dtype to use for the generated arrays.
    '''
    generator = ImageDataGenerator(
        featurewise_center=f_center,
        samplewise_center=s_center,
        featurewise_std_normalization=f_std_normalization,
        samplewise_std_normalization=s_std_normalization,
        zca_whitening=zca_whitening,
        zca_epsilon=zca_epsilon,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        fill_mode=fill_mode,
        cval=c_val,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        validation_split=validation_split,
    )
    generator.fit(image_dataset)

    augment_x = []
    batch_size = 1
    augment_y = []
    for x_batch, y_batch in generator.flow(image_dataset, labels, batch_size=batch_size):
        # for debug only
        # plt.imshow(x_batch.reshape(28, 28), cmap=plt.get_cmap('gray'))
        # plt.show()
        # exit()
        augment_x.append(x_batch)
        augment_y.append(y_batch[0])
        batch_size += 1
        if batch_size == num_of_data_to_save + 1:
            break
    save_array(np.concatenate(augment_x), np.array(augment_y), save_dir, saved_file_name)
    _show_and_save_samples(generator, image_dataset, np.array(augment_y), save_dir)


def plot_samples(x_train, y_train):
    num = 10
    images = x_train[:num]
    labels = y_train[:num]

    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()
