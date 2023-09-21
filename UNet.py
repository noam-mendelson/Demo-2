from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from torch.nn.functional import interpolate
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# ------------------------ DATA PROCESSING FUNCTIONS ------------------------
# Load images from a folder
def load_images(folder_path):
    images = [] #store loaded
    for filename in sorted(os.listdir(folder_path)): # all files and directories in the specified folder_path
        if filename.endswith(".png"): #all images png
            img_path = os.path.join(folder_path, filename) #Construct full path to image- join folder path and current filename
            img = Image.open(img_path).convert('L')  # Convert image to grayscale
            img = np.array(img) # Convert the PIL image object to a NumPy array
            images.append(img)
    return np.array(images)

# Function to preprocess data
def preprocess_data(C, root_path):
    #root_path = "/Users/noammendelson/Documents/Demo-2/keras_png_slices_data"
    root_path = "/home/Student/s4743292/keras_png_slices_data"


    #folder paths
    #input data
    X_train_folder = os.path.join(root_path, 'keras_png_slices_train') # training data
    X_val_folder = os.path.join(root_path, 'keras_png_slices_validate') # validation data- assess the performance of model during training
    X_test_folder = os.path.join(root_path, 'keras_png_slices_test') #test data- evaluate final performance of trained model after it has been trained and validated
    #output data
    y_train_folder = os.path.join(root_path, 'keras_png_slices_seg_train')
    y_val_folder = os.path.join(root_path, 'keras_png_slices_seg_validate')
    y_test_folder = os.path.join(root_path, 'keras_png_slices_seg_test')

    # Load images from folders
    X_train = load_images(X_train_folder)
    X_val = load_images(X_val_folder)
    X_test = load_images(X_test_folder)

    y_train = load_images(y_train_folder)
    y_val = load_images(y_val_folder)
    y_test = load_images(y_test_folder)

    # Normalize features to [0, 1]; target values are integers (1,2,3,0)
    #Normalisation performed on pixel values to scale down to standard range- easier for neural networks to learn from the data
    #note:range of target values were checked
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Normalize target values to be within [0, C-1] (0-3)
    y_train = (y_train / 255.0 * (C - 1)).astype(int)
    y_val = (y_val / 255.0 * (C - 1)).astype(int)
    y_test = (y_test / 255.0 * (C - 1)).astype(int)

    # Expand dimensions for X at axis=1 (second dimension- represents the number of feature maps in the input data)
    #input data in the format (num_samples, num_channels, height, width)- grayscale= 1 channel
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Number of classes
C = 4
root_path = "Demo-2/keras_png_slices_data"
preprocess_data(C, root_path)

# Preprocess data
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(C, root_path)


#convert NumPy arrays to PyTorch tensors
def numpy_to_tensor(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convert the NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).long()
    X_val_tensor = torch.tensor(X_val).float()
    y_val_tensor = torch.tensor(y_val).long()
    X_test_tensor = torch.tensor(X_test).float()
    y_test_tensor = torch.tensor(y_test).long()
    
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

# Convert data to PyTorch tensors
X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = numpy_to_tensor(X_train, y_train, X_val, y_val, X_test, y_test)


# Function to create DataLoader objects
def data_loaders(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, batch_size=32):
    batch_size = 32  

    # # Create DataLoader objects
    # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # return train_loader, val_loader, test_loader
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
    
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
# ------------------------ MODEL DEFINITIONS ------------------------

# Define the UNet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        #convolution blocks progressively reduce spatial dimensions
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Bottleneck- maintains the spatial dimensions but increases the number of channels (feature map)
        self.bottleneck = self.conv_block(256, 512, max_pooling=False)

        # Decoder (Upsampling)
        #gradually increases the spatial dimensions while decreasing the number of channels
        #reconstruct a high-resolution segmentation map from the lower-resolution feature maps generated by the encoder
        #increasing the spatial dimensions of the feature maps while reducing the number of channels
        self.upconv3 = self.upconv_block(512, 256) 
        self.upconv2 = self.upconv_block(256, 128)  
        self.upconv1 = self.upconv_block(128, 64)  

        # further refining the feature maps and adjusting the number of channels.
        self.dec3 = self.conv_block(512, 256, max_pooling=False)
        self.dec2 = self.conv_block(256, 128, max_pooling=False)
        self.dec1 = self.conv_block(128, 64, max_pooling=False)
        
        # Output Layer
        # creates a 2D convolutional layer
        self.out_conv = nn.Conv2d(64, # no. input channels to convolutional layer: previous decoder block (self.dec1) produce feature maps with 64 channels.
                                  4,  #no. output channels this convolutional layer will produce (0,1,2,3)
                                  kernel_size=1) #convolutional kernel size- performs a pointwise convolution- doesn't apply any spatial filtering but reduces no. of channels from 64 to 4.

    # Define convolutional block
    #in_channels and out_channels: no. input and output channels for the convolutional block.
    #max-pooling layer with a 2x2 kernel and a stride of 2 is added after the convolutional layers- reduces spatial dimension of feature map
    def conv_block(self, in_channels, out_channels, max_pooling=True): 
        #stores the individual layers of the block
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True) #ReLU: non-linear function- introduces non-linearity into model,for learning complex patterns and relationships in data
        ]
        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers) #sequential container for the layers defined in the layers list.
    
    # Define up-convolutional block-  used in the decoder to upsample feature maps
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True) #modify the input tensor directly to save memory
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder
        x = self.upconv3(x) #applies the up-convolutional block self.upconv3 to upsample the feature maps x.
        
        # Upsample x3 to match the spatial dimensions of x
        x3_upsampled = interpolate(x3, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        
        x = torch.cat([x, x3_upsampled], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x2_upsampled = interpolate(x2, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)  # Up-sample x2 if needed to match the spatial dimensions of x
        x = torch.cat([x, x2_upsampled], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x1_upsampled = interpolate(x1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)  # Up-sample x1 if needed to match the spatial dimensions of x
        x = torch.cat([x, x1_upsampled], dim=1)
        x = self.dec1(x)
        
        # Output Layer
        x = self.out_conv(x)
        return x

# validate the model
def validate(model, val_loader, criterion, C):
    model.eval() # Puts the model in evaluation mode
    total_val_loss = 0 #Initialises variable to accumulate the total validation loss
    dice_scores_per_label = [0.0] * C  #Initialises a list to store the Dice scores for each class- each element corresponds to a class
    total_samples = 0  # Keep track of the total number of samples
    with torch.no_grad(): #context manager that tells PyTorch not to compute gradients during this evaluation phase- memory-efficient.
        for batch in val_loader:
            inputs, labels = batch 
            outputs = model(inputs)
            loss = criterion(outputs, labels) #Calculates the loss between the model's predictions and the true labels- based on CrossEntropyLoss
            total_val_loss += loss.item()

            # Convert outputs to predicted labels
            _, preds = torch.max(outputs, 1) #predicted class labels by taking the index of the class with the highest probability for each pixel.
            preds = preds.cpu().numpy().flatten() #Converts the predicted labels to a NumPy array and flattens- computational ease
            labels = labels.cpu().numpy().flatten() # Converts the true labels to a NumPy array and flattens it.

            # Calculate F1 score (Dice Similarity Coefficient) for each label
            for label in range(C):
                dice_score = f1_score(labels == label, preds == label)
                dice_scores_per_label[label] += dice_score

            total_samples += 1  # Update the total number of samples

    # Average the Dice scores and the loss
    avg_val_loss = total_val_loss / total_samples
    avg_dice_scores = [score / total_samples for score in dice_scores_per_label]

    return avg_val_loss, avg_dice_scores

# ------------------------ MODEL DEFINITIONS ------------------------
model = UNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")  # Print the loss after each epoch
    
    val_loss, avg_dice_scores = validate(model, val_loader, criterion, C)
    print(f"Validation loss after epoch {epoch + 1}: {val_loss}")
    print(f"Avg DSC per label after epoch {epoch + 1}: {avg_dice_scores}")
    print(f"Epoch {epoch + 1} completed.")

# Function to display sample images
def display_sample(X, y_true, y_pred, class_map):
    """Display a sample image, its ground truth and its predicted mask."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    plt.imshow(y_true[0], cmap='tab10', vmin=0, vmax=len(class_map)-1)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(y_pred[0], cmap='tab10', vmin=0, vmax=len(class_map)-1)
    plt.axis('off')

    plt.show()
