from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')  # Convert image to grayscale
            img = np.array(img)
            images.append(img)
    print(f"Loaded {len(images)} images from {folder_path}")  # Debugging line
    return np.array(images)

def preprocess_data():
    root_path = '/Users/noammendelson/Documents/Demo-2/keras_png_slices_data/'

    # Define folder paths
    X_train_folder = os.path.join(root_path, 'keras_png_slices_train')
    X_val_folder = os.path.join(root_path, 'keras_png_slices_validate')
    X_test_folder = os.path.join(root_path, 'keras_png_slices_test')

    y_train_folder = os.path.join(root_path, 'keras_png_slices_seg_train')
    y_val_folder = os.path.join(root_path, 'keras_png_slices_seg_validate')
    y_test_folder = os.path.join(root_path, 'keras_png_slices_seg_test')

    # Load images from folders
    X_train = load_images_from_folder(X_train_folder)
    X_val = load_images_from_folder(X_val_folder)
    X_test = load_images_from_folder(X_test_folder)

    y_train = load_images_from_folder(y_train_folder)
    y_val = load_images_from_folder(y_val_folder)
    y_test = load_images_from_folder(y_test_folder)

    # Normalize features to [0, 1] and assume labels are integers
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = y_train.astype('int')
    y_val = y_val.astype('int')
    y_test = y_test.astype('int')

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()







    # Encoder (Downsampling)
    # Add a series of Conv2D, Activation, and MaxPooling layers
    # can also add BatchNormalization and Dropout layers 
    
    # Bottleneck
    # Add a series of Conv2D and Activation layers
    #can also add BatchNormalization and Dropout layers 
    
    # Decoder (Upsampling)
    # Add a series of Conv2D, Activation, and UpSampling2D or Conv2DTranspose layers
    # can also add BatchNormalization and Dropout layers 
    
    # Output Layer
    # Add a Conv2D layer with softmax activation for segmentation classes
    
    # Compile Model
    # Use an appropriate optimizer and loss function for segmentation tasks
    # Dice loss or binary cross-entropy 