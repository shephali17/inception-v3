import numpy as np
import pandas as pd
import cv2, os
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from torchsummary import summary
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test = 1

# Defining the Directories
train_data_dir = "/home/shephali/Desktop/InceptionV2/train"
test_data_dir = "/home/shephali/Desktop/InceptionV2/test"
pred_data_dir = "/home/shephali/Desktop/InceptionV2/prediction"

# Checking the number of Train Images
for i in os.listdir(train_data_dir):
    new_loc = os.path.join(train_data_dir, i)
    new = new_loc + '/*.jpg'
    images = glob(new)
    print(f'{i}:', len(images))

# Checking the number of Test Images
for i in os.listdir(test_data_dir):
    new_loc = os.path.join(test_data_dir, i)
    new = new_loc + '/*.jpg'
    images = glob(new)
    print(f'{i}:', len(images))

# Getting the classes and their meaning in a dictionary
classes = os.listdir(train_data_dir)
classes = {k: v for k, v in enumerate(sorted(classes))}
print(classes)

# Performing the Image Transformation and Data Augmentation on the 
# train dataset and transformation on Validation Dataset

# Convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to RGB
    transforms.Resize((299, 299)),  # Resize to the size expected by Inception v3
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Augmentation on test images not needed
transform_tests = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to RGB
    transforms.Resize((299, 299)),  # Resize to the size expected by Inception v3
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Using ImageFolder function for fetching data from directory
train_data = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_data_dir, transform=transform_tests)

training_data = DataLoader(train_data, batch_size=32, drop_last=True, shuffle=True, num_workers=2)
validation_data = DataLoader(test_data, batch_size=32, drop_last=True, shuffle=True, num_workers=2)

model = models.inception_v3(pretrained=True)

# Defining the Model Function.
# Lets freeze all layers and change just a few layers to match our requirements
def get_model():
    model = models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freezing all the layers and changing only the below layers
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 6)
    )
    model.aux_logits = False
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_fn, optimizer

input_shape = (3, 299, 299)
summary(model.to(device), input_shape)

# Creating the Utility function to get the Losses and Accuracies for Train and Validation Dataset 
def train_batch(x, y, model, opt, loss_fn):
    model.train()
    output = model(x)
    batch_loss = loss_fn(output, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x, y, model):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()

# Initializing the Model, Loss Function, and Optimizer to a Variable
model, loss_fn, optimizer = get_model()

# Start the Model Training and save the Losses and Accuracies of Both train and validation
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(10):
    print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    for ix, batch in enumerate(training_data):
        x, y = batch
        x, y = x.to(device), y.to(device)
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
        train_epoch_losses.append(batch_loss)        
    train_epoch_loss = np.array(train_epoch_losses).mean()
    train_epoch_accuracy = np.mean(train_epoch_accuracies)        
    print('Epoch:', epoch, 'Train Loss:', train_epoch_loss, 'Train Accuracy:', train_epoch_accuracy)

    val_epoch_losses, val_epoch_accuracies = [], []
    for ix, batch in enumerate(validation_data):
        x, y = batch
        x, y = x.to(device), y.to(device)
        val_is_correct = accuracy(x, y, model)
        validation_loss = val_loss(x, y, model)
        val_epoch_accuracies.extend(val_is_correct)
        val_epoch_losses.append(validation_loss)
    val_epoch_loss = np.array(val_epoch_losses).mean()
    val_epoch_accuracy = np.mean(val_epoch_accuracies)
    
    print('Epoch:', epoch, 'Validation Loss:', val_epoch_loss, 'Validation Accuracy:', val_epoch_accuracy)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_accuracy)

# Define the function to predict the images from the prediction set
def pred_class(img):
    img_tens = transform_tests(img)  # Transform the image
    img_im = img_tens.unsqueeze(0).to(device)  # Change image format to (1, 3, 299, 299)
    with torch.no_grad():
        out = model(img_im)
    index = out.cpu().numpy().argmax()  # Get the predicted class index
    return index

# Get the location of all the prediction files
pred_files = [os.path.join(pred_data_dir, f) for f in os.listdir(pred_data_dir)]
print(pred_files[:10])

# Prediction Results
model.eval()

plt.figure(figsize=(20, 20))
for i, image_path in enumerate(pred_files):
    if i > 24:
        break
    img = Image.open(image_path)
    index = pred_class(img)
    plt.subplot(5, 5, i + 1)
    plt.title(classes[index])
    plt.axis('off')
    plt.imshow(img)
plt.show()
