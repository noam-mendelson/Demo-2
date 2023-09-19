from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset 

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

def numpy_to_tensor(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convert the NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).long()
    X_val_tensor = torch.tensor(X_val).float()
    y_val_tensor = torch.tensor(y_val).long()
    X_test_tensor = torch.tensor(X_test).float()
    y_test_tensor = torch.tensor(y_test).long()
    
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

def data_loaders(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, batch_size=32):
    # Create DataLoader objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = numpy_to_tensor(X_train, y_train, X_val, y_val, X_test, y_test)
train_loader, val_loader, test_loader = data_loaders(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor) 

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder (Upsampling)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        # Output Layer
        self.out_conv = nn.Conv2d(64, 4, kernel_size=1)  #4 classes for segments
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder
        x = self.dec3(x)
        x += x3
        x = self.dec2(x)
        x += x2
        x = self.dec1(x)
        x += x1
        
        # Output layer
        x = self.out_conv(x)
        
        return x



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