import torch 
import torch.nn as nn
from models.vgg import vgg_16
from utils.utils import data_loader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available:", device)

# Hyperparameters
batch_size = 128
num_epochs = 32

# Load dataset
data_dir = "/media/data/Project_Only/Pytorch_Tutorials/data/RAFDB/aligned"
label_file = "/media/data/Project_Only/Pytorch_Tutorials/data/RAFDB/list_patition_label.txt"
train_loader, valid_loader = data_loader(data_dir=data_dir, label_file=label_file, batch_size=batch_size)

# Define model
model = vgg_16(num_classes=11).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.005, momentum = 0.9) 

# Train the model
writer = SummaryWriter("runs/RAFDB")
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Write log from epoch
        writer.add_scalar("Loss", loss, epoch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        # Write accuracy to TensorBoard
        writer.add_scalar('Accuracy', (100 * correct / total), epoch)

        print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total)) 

# Close the writer
writer.close()

# Save the model checkpoint
save_dir = "./weights"
torch.save(model.state_dict(), os.path.join(save_dir, 'vgg_16.pt'))