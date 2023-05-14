import torch 
from utils.data_loader import data_loader
from models.resnet import ResNet, ResidualBlock
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available:", device)

# Define model Resnet
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
model.load_state_dict(torch.load('./weights/resnet_50.pt'))
print("Load model Resnet sucessfully")

# Define data transformations for augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])

# Load data 
img_path = "/media/data/Project_Only/MLOps/face-rec-avenger/data/01_raw/robert_downey_jr/test/robert_downey_jr36.png"
input_img = Image.open(img_path)

# do transformations
input_data = transform(input_img).to(device)

# Model output
input_data = torch.unsqueeze(input_data, 0)
output = model(input_data)

# Convert output to probabilities using softmax
softmax = torch.nn.Softmax(dim=1)
probs = softmax(output)

# Get the predicted class label with the highest probability
predicted_class = torch.argmax(probs)

# Print the predicted class label
print("Predicted class label:", predicted_class.item())