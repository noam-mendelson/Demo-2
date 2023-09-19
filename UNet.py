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
    return np.array(images)

def preprocess_data():
    X_folder = '/Users/noammendelson/Documents/Demo-2/keras_png_slices_data/X_folder'
    y_folder = '/Users/noammendelson/Documents/Demo-2/keras_png_slices_data/y_folder'

    X = load_images_from_folder(X_folder)
    y = load_images_from_folder(y_folder)

    # Normalize your data to [0, 1]
    X = X.astype('float32') / 255.0
    y = y.astype('float32') / 255.0

    # Data splitting
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_train, X_val, y_train, y_val






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