import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import shutil
import random
import os


def remove_data(input_dir):
    
    # Iterate through files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            os.remove(os.path.join(input_dir, filename))

    return print("Successfully removed data!")


def split_data(input_dir, output_dir, label_file): 

    # Create train and test directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create directories for each label
    with open(label_file, "r") as f:
        labels = f.read().splitlines()

    for label in labels:
        image_name, label_name = label.split(" ")
        os.makedirs(os.path.join(train_dir, label_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label_name), exist_ok=True)

        # Iterate through files in input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                filename_new = filename.replace("_aligned", "")

                if filename_new == image_name:
                    # Determine whether to put image in train or test directory
                    if "train" in filename_new:
                        output_subdir = train_dir
                    else:
                        output_subdir = test_dir

                    # print("A:", image_name, label_name)
                    # print("B:", filename_new, filename)
                    # print("C:", output_subdir, label_name)
                    # print("FINAL:", os.path.join(input_dir, filename), os.path.join(output_subdir, label_name))

                    # Move file to appropriate directory
                    shutil.move(os.path.join(input_dir, filename), os.path.join(output_subdir, label_name))
                    break
        # break
    print("Successfully created train and test directories!")


def data_loader(data_dir, label_file, batch_size, shuffle=True):
    
    # Define data transformations for augmentation
    transform_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Define data transformations for validation/testing
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load image labels from file
    with open(label_file, "r") as f:
        labels = f.read().splitlines()
    
    # Create a dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform_aug)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform_val)

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("Successfully load data!")
    return train_loader, test_loader


def main():
    data_dir = "/media/data/Project_Only/FER_Pytorch/data/aligned"
    label_file = "/media/data/Project_Only/FER_Pytorch/data/list_patition_label.txt"
    split_data(data_dir, data_dir, label_file)
    # remove_data(data_dir)


if __name__ == "__main__":
    main()