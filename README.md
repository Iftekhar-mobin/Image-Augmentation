# Handwritten Digits Augmentation, Augmented digit Sequence Generator for classification using MNIST dataset

### Show your support

Give a ⭐ if this project was helpful in any way!

## Purpose

The goal of this project is to generate images representing sequences of numbers, for data augmentation purposes.
These images would be used to train classifiers and generative deep learning models.  Save examples of generated images 
will be used for inspecting the characteristics of the generated images and for inspecting the trained models behaviours.

## Constraints
1. Download digit dataset from (http://yann.lecun.com/exdb/mnist/),
2. Files are processed and image data is Normalized. 
3. The digits are stacked horizontally. e.g. [2, 9, 3 , 4] input given by user.
4. Spacing between digits follow a uniform distribution. User can **specify width**. 
5. Spacing range determined by user **{min spacing, max spacing}**. 
6. Each digit in the generated sequence is chosen randomly from MNIST dataset.
7. The width of the output image in pixels is specified by the user; height is 28 pixels (i.e. identical to MNIST digits). 
8. code is given an API to call it easily from any Third party library/interface or code.

![download (2)](https://user-images.githubusercontent.com/54829611/115188378-cc628980-a11f-11eb-8579-c5210df6580e.png)

## Solution
The solution is distributed in following steps:
### step 01: install all the libraries and dependencies
1. The project support
    > python >= 3.6
2. In case of matplotlib visualization problem    
    > sudo apt-get install python3-tk 
3. install all the libraries and dependencies
    > pip install -r requirements.txt 
### step 02: load dataset and normalize
1. Used mlxtend library to load dataset
    > mlxtend==0.18.0
2. converted data to 32 bit floating point
    > images = images.reshape(10000, 28, 28).astype('float32')
### step 03: create a mapping with class label 
1. For an example, marking position 7 for value 7, of index 0 in class labels     
    ```
    For i=0 it will be [[], [], [], [], [], [], [], [0], [], []]
    dataset_labels[0] = 7 
    ```
### step 04: Automatic space is calculated to fit in Users' provided width

    auto_space = (image_width - image_height * number_of_digits in sequence) / number_of_space required 
    space_generator(length_input_sequence, image_width, image_height, spacing_range)

![0288502623](https://user-images.githubusercontent.com/54829611/115187760-d89a1700-a11e-11eb-8a8d-3c62df7f9a8b.png)
### step 05: generating horizontal image stacking one after another
    
    generate_image(dataset_images, label_mapping, input_sequence, spacing_range, image_width, image_height=28)
    
### step 06: generate random images of fixed length
    generate_random_sequence(num_samples, seq_len, dataset_images, label_mapping, spacing)

### step 07: Used keras ImageDataGenerator to augment the MNIST dataset Images
        
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

### step 08: Generated data is saved, visualized and step 1~6 step is repeated
    There is a jupyter notebook is attached to analyze every step in details

### step 09: Run the following API to generate desired number of samples for Training/Testing model

## API: The API is very straight forward
1. There is a **constants.py** file to contain all the required constant values
    ```
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
    ```
![download](https://user-images.githubusercontent.com/54829611/115188123-6a098900-a11f-11eb-95bb-033701cf4161.png)
2. Run the **api.py** to see generated files 
3. The parameters are self-explanatory
```
# To generate only one sequence
driver.horizontal_seq_gen(
    input_sequence=[2, 4, 1, 7],
    space_range={'min': 0, 'max': 5},
    image_width=200
)
```
```
# To generate bulk without augmenting source dataset
driver.generate_random_sequence(
    num_samples=10,
    seq_len=10,
    space_range={'min': 0, 'max': 10},
    image_width=200
)
```
```
# Data augmentation and save augmented data
driver.augment_dataset(
    num_of_data_to_save=10000,
    zca_whitening=False,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
)
```

```
########################################################################
#    This Step is generate horizontal Images after augmentation        #
########################################################################
# Just make sure before running individually run **driver.augment_dataset()** first
augmented = AugmentedSeqGen()
augmented.visualize_generated_aug_dataset()
```
![download (1)](https://user-images.githubusercontent.com/54829611/115188268-a50bbc80-a11f-11eb-8fa1-db7bea91b9db.png)

```
# To generate bulk after augmenting source dataset with keras
augmented.generate_random_sequence_with_keras_data(
    see_image=True,
    num_samples=10,
    seq_len=10,
    space_range={'min': 0, 'max': 10},
    image_width=200
)
```

where:
 
- `see_image` False data is normalized [0-1] range. Hence, image is not visible. if True normalized to [0-255]  
- `space_range` is a dict format containing the minimum and maximum spacing (in pixels) between digits. 
- `image_width` specifies the width of the image in pixels.
- `num_samples` is the number of samples of image dataset to generate from/to dataset.
- `seq_len` is the length of the sequence for each image in the image dataset. sequence length 2, e.g. [1, 7], 

The generated image is saved in the directory 
- `aug_gen_data` image sequence samples are generated from MNIST dataset  
- `keras_data` contain only individual digit samples of augmented data. 
- `keras_hor_images` generate augmented dataset in this directory. 
  
The file name specifies the labels of the image sequence within 
- `aug_gen_data` - also contain compressed numpy array for loading/reuse 
- `keras_hor_images` - also contain compressed numpy array for loading/reuse 

### Show your support
Give a ⭐ if this project was helpful in any way!

