import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, class0_dir, class1_dir, transform=None):
        self.class0_dir = class0_dir
        self.class1_dir = class1_dir
        self.transform = transform

        self.class0_images = [
            os.path.join(self.class0_dir, fname)
            for fname in os.listdir(self.class0_dir)
            if fname.endswith('.png')
        ]
        self.class1_images = [
            os.path.join(self.class1_dir, fname)
            for fname in os.listdir(self.class1_dir)
            if fname.endswith('.png')
        ]

        self.images = self.class0_images + self.class1_images
        self.labels = [0] * len(self.class0_images) + [1] * len(self.class1_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Read 16-bit single-channel image and convert to 8-bit
        image = Image.open(img_path)
        image = image.convert('I')  # Preserve 16-bit depth
        image = image.point(lambda x: x * (1.0 / 256)).convert('L')  # Scale to 8-bit

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 1)
            # Not using Sigmoid, using BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # Output logits

# Define global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['CNN0', 'CNN1']

# Initialize the model
model = CNNModel().to(device)

def predict_image(image_path, model, device, class_names):
    """
    Predicts the class of a single image.

    Args:
        image_path (str): Path to the image to be predicted.
        model (nn.Module): The model with loaded weights.
        device (torch.device): The computation device.
        class_names (list): List of class names.

    Returns:
        None
    """
    image = Image.open(image_path)
    image = image.convert('I')  # Preserve 16-bit depth
    image = image.point(lambda x: x * (1.0 / 256)).convert('L')  # Scale to 8-bit

    # Preprocessing identical to training (without random data augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Mean and std for single-channel images
    ])

    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)  # Output logits
        probability = torch.sigmoid(output).item()  # Apply Sigmoid function
        prediction = 1 if probability > 0.5 else 0
        print(f"Prediction Result: Class {class_names[prediction]}, Probability: {probability:.4f}")

def main():
    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),      # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Mean and std for single-channel images
    ])

    # Create dataset
    dataset = CustomDataset(
        class0_dir='D:/Code/phys134/CNN0',
        class1_dir='D:/Code/phys134/CNN1',
        transform=data_transforms
    )

    # Split dataset into training and validation sets while maintaining class proportions
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.1,
        shuffle=True,
        stratify=dataset.labels,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create data loaders
    batch_size = 16  # Adjust based on GPU memory
    num_workers = 0  # Set to 0 to avoid multiprocessing issues on Windows

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Train the model
    num_epochs = 30

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure label shape consistency

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Output logits
                    probs = torch.sigmoid(outputs)  # Apply Sigmoid
                    preds = (probs > 0.5).float()
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

        # Update learning rate
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("Model saved as 'cnn_model.pth'")

    # Visualize training process
    epochs_range = range(num_epochs)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Example Prediction
    test_image_path = 'D:/Code/phys134/test_image.png'  # Replace with your test image path
    if os.path.exists(test_image_path):
        # Load weights and make prediction
        model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
        predict_image(test_image_path, model, device, class_names)
    else:
        print(f"Test image path {test_image_path} does not exist.")

if __name__ == "__main__":
    main()
