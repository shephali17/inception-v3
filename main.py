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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Defining the Directories
train_data_dir = "/home/shephali/Desktop/inception-v3/train"
test_data_dir = "/home/shephali/Desktop/inception-v3/test"
pred_data_dir = "/home/shephali/Desktop/inception-v3/prediction"

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

# Performing the Image Transformation and Data Augmentation on the train dataset and transformation on Validation Dataset

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
# Let's freeze all layers and change just a few layers to match our requirements
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
        nn.Linear(128, len(classes))
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
    opt.step()
    opt.zero_grad()
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

# Early stopping and model saving
best_val_loss = float('inf')
best_model_wts = None
patience = 10
trigger_times = 0

# Initializing the Model, Loss Function, and Optimizer to a Variable
model, loss_fn, optimizer = get_model()

# Start the Model Training and save the Losses and Accuracies of Both train and validation
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

epochs = 100

for epoch in range(epochs):
    print(f'Epoch {epoch}/{epochs-1}')
    print('-' * 10)

    train_epoch_losses, train_epoch_accuracies = [], []
    val_epoch_losses, val_epoch_accuracies = [], []

    # Training phase
    model.train()
    for batch in tqdm(training_data, desc="Training"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        train_epoch_losses.append(batch_loss)
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)

    # Validation phase
    model.eval()
    for batch in tqdm(validation_data, desc="Validation"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        validation_loss = val_loss(x, y, model)
        val_epoch_losses.append(validation_loss)
        is_correct = accuracy(x, y, model)
        val_epoch_accuracies.extend(is_correct)

    train_epoch_loss = np.mean(train_epoch_losses)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)
    val_epoch_loss = np.mean(val_epoch_losses)
    val_epoch_accuracy = np.mean(val_epoch_accuracies)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_accuracy)

    print(f'Train Loss: {train_epoch_loss:.4f} Train Accuracy: {train_epoch_accuracy:.4f}')
    print(f'Validation Loss: {val_epoch_loss:.4f} Validation Accuracy: {val_epoch_accuracy:.4f}')

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        best_model_wts = model.state_dict()
        trigger_times = 0
    else:
        trigger_times += 1

    if trigger_times >= patience:
        print('Early stopping!')
        break

# Load the best model weights
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

# Plotting the training and validation loss and accuracy
epochs_range = range(len(train_losses))
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.show()

# Define the function to predict the images from the prediction set
def pred_class(img):
    img_tens = transform_tests(img)  # Transform the image
    img_im = img_tens.unsqueeze(0).to(device)  # Change image format to (1, 3, 299, 299)
    with torch.no_grad():
        out = model(img_im)
    index = out.cpu().numpy().argmax()  # Get the predicted class index
    return index

# Get the location of all the prediction files
pred_files = []
for root, dirs, files in os.walk(pred_data_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            pred_files.append(os.path.join(root, file))

print(pred_files[:10])

# Prediction of the class for the images in prediction folder


output = [pred_class(Image.open(f)) for f in pred_files]

# Add more detailed evaluation metrics
y_true = []
y_pred = []

model.eval()
for batch in tqdm(validation_data, desc="Evaluation"):
    x, y = batch
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
    y_true.extend(y.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

report = classification_report(y_true, y_pred, target_names=classes.values())
print("Classification Report:\n", report)

f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')

print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')