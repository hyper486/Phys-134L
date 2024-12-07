# grad_cam_visualization.py

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from statsmodels.stats.contingency_tables import mcnemar
import warnings
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Suppress any potential warnings for cleaner output
warnings.filterwarnings('ignore')



# Define the device globally
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define class names globally based on your dataset
class_names = ['Globular', 'Open']  # Original class names

# Mapping for plot labels
label_mapping = {
    0: 'Globular Cluster',
    1: 'Open Cluster'
}


# Custom CNN Model Definition
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
            nn.Linear(256 * 14 * 14, 1)  # Adjust based on input image size
            # Not using Sigmoid, as we'll apply it during evaluation
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # Outputs logits


# Function to Load Models
def load_model(model_path, model_type='resnet101'):
    """
    Load the trained model.

    Args:
        model_path (str): Path to the model weights file.
        model_type (str): Type of the model ('resnet101' or 'cnn').

    Returns:
        model (nn.Module): The loaded model ready for evaluation.
    """
    if model_type == 'resnet101':
        # Initialize ResNet-101 model without pre-trained weights
        model = models.resnet101(pretrained=False)
        
        # Modify the first convolutional layer to accept single-channel images
        model.conv1 = nn.Conv2d(
            in_channels=1,      # Change from 3 (RGB) to 1 (Grayscale)
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Reinitialize the weights of the modified conv1 layer
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Modify the fully connected layer to output a single value for binary classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif model_type == 'cnn':
        # Initialize your custom CNN model
        model = CNNModel()
    else:
        raise ValueError("Unsupported model type. Choose 'resnet101' or 'cnn'.")
    
    # Load the saved state_dict (weights)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move the model to the specified device
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model



# Preprocessing Function
def preprocess_image(image_path):
    """
    Preprocess the input image for model prediction.

    Args:
        image_path (str): Path to the input image.

    Returns:
        image_tensor (torch.Tensor): The preprocessed image tensor.
    """
    try:
        # Open the image and convert to 16-bit grayscale
        image = Image.open(image_path).convert('I')
        # Scale the image to 8-bit grayscale
        image = image.point(lambda x: x * (1./256)).convert('L')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        # Return a blank tensor if the image cannot be opened
        return torch.zeros(1, 224, 224).unsqueeze(0).to(device)
    
    # Define the preprocessing transformations (should match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Resize to 224x224 pixels
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize([0.5], [0.5])   # Normalize with mean and std for single-channel
    ])
    
    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)


# Custom Dataset Class
class TestDataset(Dataset):
    """
    Custom dataset class for loading test images.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset with image paths and labels.

        Args:
            image_paths (list): List of image file paths.
            labels (list): Corresponding list of labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Retrieve the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image tensor and label is the class label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Open and preprocess the image
            image = Image.open(img_path).convert('I')  # Convert to 16-bit grayscale
            image = image.point(lambda x: x * (1./256)).convert('L')  # Scale to 8-bit
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            # Create a blank image if there's an error
            image = Image.new('L', (224, 224))
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image, label


# Utility Function to Check Image Files
def is_image_file(filename):
    """
    Check if a file is an image based on its extension.

    Args:
        filename (str): Name of the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return filename.lower().endswith(image_extensions)


# Evaluation Function
def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test dataset and compute performance metrics.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        dict: Dictionary containing all performance metrics.
    """
    all_preds = []
    all_probs = []
    all_labels = []
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Flatten the lists and convert to integers
    all_preds = [int(p[0]) for p in all_preds]
    all_probs = [p[0] for p in all_probs]
    all_labels = [int(l[0]) for l in all_labels]
    
    # Compute performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # Generate classification report
    class_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    # Store metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'classification_report': class_report,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs
    }
    
    return metrics


# Plotting Functions
def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix using seaborn heatmap.

    Args:
        conf_matrix (ndarray): Confusion matrix array.
        class_names (list): List of class names.

    Returns:
        None
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(all_labels, all_probs, roc_auc):
    """
    Plot the ROC curve.

    Args:
        all_labels (list): True labels.
        all_probs (list): Predicted probabilities.
        roc_auc (float): ROC AUC score.

    Returns:
        None
    """
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_misclassified_images(all_labels, all_preds, test_paths, label_mapping, num_images=5):
    """
    Plot a specified number of misclassified images.

    Args:
        all_labels (list): True labels.
        all_preds (list): Predicted labels.
        test_paths (list): List of image paths.
        label_mapping (dict): Mapping from label indices to class names.
        num_images (int): Number of misclassified images to display.

    Returns:
        None
    """
    misclassified = [i for i, (true, pred) in enumerate(zip(all_labels, all_preds)) if true != pred]
    if not misclassified:
        print("No misclassified images to display.")
        return

    num_images = min(num_images, len(misclassified))
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        idx = misclassified[i]
        img_path = test_paths[idx]
        try:
            image = Image.open(img_path).convert('I').point(lambda x: x * (1./256)).convert('L')
            image = image.resize((224, 224))
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            image = Image.new('L', (224, 224))
        plt.subplot(1, num_images, i+1)
        plt.imshow(image, cmap='gray')
        # Map labels using label_mapping
        true_label = label_mapping.get(all_labels[idx], 'Unknown')
        pred_label = label_mapping.get(all_preds[idx], 'Unknown')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.show()


# McNemar Test Function
def perform_mcnemar_test(preds_a, preds_b):
    """
    Perform the McNemar Test between two sets of predictions.

    Args:
        preds_a (list): Predictions from Model A.
        preds_b (list): Predictions from Model B.

    Returns:
        None
    """
    # Ensure both prediction lists are of the same length
    assert len(preds_a) == len(preds_b), "Prediction lists must be of the same length."
    
    # Initialize counts
    n00 = n01 = n10 = n11 = 0
    
    for a, b in zip(preds_a, preds_b):
        if a == 1 and b == 1:
            n11 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
        else:
            n00 += 1
    
    print("Contingency Table:")
    print(f"n00 (Both Correct): {n00}")
    print(f"n01 (Model A Incorrect, Model B Correct): {n01}")
    print(f"n10 (Model A Correct, Model B Incorrect): {n10}")
    print(f"n11 (Both Incorrect): {n11}\n")
    
    # Create contingency table for McNemar Test
    table = [[n00, n01],
             [n10, n11]]
    
    # Perform McNemar Test
    result = mcnemar(table, exact=True)
    
    print("McNemar Test Results:")
    print(f"Statistic: {result.statistic}")
    print(f"P-value: {result.pvalue}")
    
    # Interpret the result
    alpha = 0.05
    if result.pvalue < alpha:
        print("Result: Significant difference between the two models (reject H0).")
    else:
        print("Result: No significant difference between the two models (fail to reject H0).")


# Grad-CAM Visualization Function
def visualize_grad_cam(model, test_paths, all_labels, all_preds, target_layer_idx=9, num_images=1):
    """
    Visualize Grad-CAM for misclassified images.

    Args:
        model (nn.Module): Trained model.
        test_paths (list): List of image paths.
        all_labels (list): True labels.
        all_preds (list): Predicted labels.
        target_layer_idx (int): Index of the target convolutional layer in model.features.
        num_images (int): Number of misclassified images to visualize.

    Returns:
        None
    """
    # Identify misclassified samples
    errors = [i for i, (true, pred) in enumerate(zip(all_labels, all_preds)) if true == pred]
    
    if not errors:
        print("No misclassified samples to visualize with Grad-CAM.")
        return
    
    for idx in range(min(num_images, len(errors))):
        error_idx = errors[idx]
        img_path = test_paths[error_idx]
        
        # Load and preprocess the image
        try:
            image = Image.open(img_path).convert('I').point(lambda x: x * (1./256)).convert('L')
            image = image.resize((224, 224))
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            image = Image.new('L', (224, 224))
        
        input_tensor = preprocess_image(img_path)
        
        # Initialize Grad-CAM
        target_layer = model.features[target_layer_idx]  # Adjust based on your model
        grad_cam = GradCAM(model=model, target_layers=[target_layer])  # Removed use_cuda
        
        # Create a target for class 1 (since binary classification with single logit)
        targets = [ClassifierOutputTarget(0)]  # The single logit corresponds to class 1
        
        # Generate the heatmap
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Prepare the image for visualization
        img_np = np.array(image.convert('RGB')) / 255.0  # Normalize to [0,1]
        heatmap = grayscale_cam
        cam_image = show_cam_on_image(img_np, heatmap, use_rgb=True)
        
        # Plot the original image and Grad-CAM overlay
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.title('Grad-CAM')
        plt.axis('off')
        
        plt.show()


def main():
    """
    Main function to evaluate both models and perform the McNemar Test.
    """
    # Specify model paths
    resnet101_model_path = 'resnet101_model.pth'  # Path to ResNet-101 model
    cnn_model_path = 'cnn_model.pth'              # Path to custom CNN model
    
    # Specify test directory
    test_dir = 'D:/Code/phys134/test'             # Path to test dataset directory
    
    # Check if model files exist
    if not os.path.exists(resnet101_model_path):
        print(f"Error: Model file '{resnet101_model_path}' does not exist.")
        return
    if not os.path.exists(cnn_model_path):
        print(f"Error: Model file '{cnn_model_path}' does not exist.")
        return
    
    # Load both models
    print("Loading ResNet-101 model...")
    resnet101_model = load_model(resnet101_model_path, model_type='resnet101')
    
    print("Loading Custom CNN model...")
    cnn_model = load_model(cnn_model_path, model_type='cnn')
    
    # Prepare test image paths and labels
    test_paths = []
    test_labels = []
    
    # Assume test directory has subdirectories for each class
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory '{class_dir}' does not exist. Skipping.")
            continue
        for fname in os.listdir(class_dir):
            if is_image_file(fname):
                test_paths.append(os.path.join(class_dir, fname))
                test_labels.append(class_names.index(class_name))  # Assign label based on class index
    
    # Check if there are any test images
    if not test_paths:
        print(f"Error: No valid image files found in directory '{test_dir}'.")
        return
    
    # Define the same preprocessing transformations used during training
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),       # Resize images to 224x224
        transforms.ToTensor(),               # Convert images to PyTorch tensors
        transforms.Normalize([0.5], [0.5])   # Normalize with mean and std for single-channel
    ])
    
    # Create the test dataset and data loader
    test_dataset = TestDataset(test_paths, test_labels, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Function to get predictions from a model
    def get_predictions(model, test_loader):
        """
        Get predictions from the model on the test dataset.

        Args:
            model (nn.Module): The trained model.
            test_loader (DataLoader): DataLoader for the test dataset.

        Returns:
            tuple: (all_preds, all_probs, all_labels)
        """
        all_preds = []
        all_probs = []
        all_labels = []
        
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)  # Ensure labels have shape [batch_size, 1]
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Flatten the lists and convert to integers
        all_preds = [int(p[0]) for p in all_preds]
        all_probs = [p[0] for p in all_probs]
        all_labels = [int(l[0]) for l in all_labels]
        
        return all_preds, all_probs, all_labels
    
    # Get predictions from both models
    print("\nGenerating predictions from ResNet-101 model...")
    resnet101_preds, resnet101_probs, resnet101_labels = get_predictions(resnet101_model, test_loader)
    
    print("Generating predictions from Custom CNN model...")
    cnn_preds, cnn_probs, cnn_labels = get_predictions(cnn_model, test_loader)
    
    # Ensure that labels from both models are the same
    assert resnet101_labels == cnn_labels, "Mismatch in labels between models."
    
    # Compute and print performance metrics for both models
    def print_metrics(model_name, preds, probs, labels):
        """
        Compute and print performance metrics.

        Args:
            model_name (str): Name of the model.
            preds (list): Predicted labels.
            probs (list): Predicted probabilities.
            labels (list): True labels.

        Returns:
            dict: Dictionary containing all performance metrics.
        """
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        conf_matrix = confusion_matrix(labels, preds)
        roc_auc = roc_auc_score(labels, probs)
        class_report = classification_report(labels, preds, target_names=class_names, zero_division=0)
        
        print(f"\n--- {model_name} Performance Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"\nClassification Report:\n{class_report}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'classification_report': class_report,
            'all_labels': labels,
            'all_preds': preds,
            'all_probs': probs
        }
    
    # Evaluate both models
    resnet101_metrics = print_metrics("ResNet-101", resnet101_preds, resnet101_probs, resnet101_labels)
    cnn_metrics = print_metrics("Custom CNN", cnn_preds, cnn_probs, cnn_labels)
    
    # Plot performance metrics for ResNet-101
    plot_confusion_matrix(resnet101_metrics['confusion_matrix'], class_names)
    plot_roc_curve(resnet101_metrics['all_labels'], resnet101_metrics['all_probs'], resnet101_metrics['roc_auc'])
    
    # Plot performance metrics for Custom CNN
    plot_confusion_matrix(cnn_metrics['confusion_matrix'], class_names)
    plot_roc_curve(cnn_metrics['all_labels'], cnn_metrics['all_probs'], cnn_metrics['roc_auc'])
    
    # Perform McNemar Test
    print("\nPerforming McNemar Test between ResNet-101 and Custom CNN...")
    perform_mcnemar_test(resnet101_preds, cnn_preds)
    
    # Optionally, plot misclassified images for both models
    print("\nPlotting misclassified images for ResNet-101...")
    plot_misclassified_images(
        resnet101_metrics['all_labels'],
        resnet101_metrics['all_preds'],
        test_paths,
        label_mapping,
        num_images=5
    )
    
    print("Plotting misclassified images for Custom CNN...")
    plot_misclassified_images(
        cnn_metrics['all_labels'],
        cnn_metrics['all_preds'],
        test_paths,
        label_mapping,
        num_images=5
    )
    

    # Grad-CAM Visualization
    print("\nVisualizing Grad-CAM for misclassified images...")
    visualize_grad_cam(
        model=cnn_model,                # Choose which model to visualize ('resnet101_model' or 'cnn_model')
        test_paths=test_paths,
        all_labels=cnn_metrics['all_labels'],
        all_preds=cnn_metrics['all_preds'],
        target_layer_idx=9,             # Adjust based on your model's architecture
        num_images=30                    # Number of misclassified images to visualize
    )

    # print("\nVisualizing Grad-CAM for misclassified images...")
    # visualize_grad_cam(
    #     model=resnet101_model,                # Choose which model to visualize ('resnet101_model' or 'cnn_model')
    #     test_paths=test_paths,
    #     all_labels=cnn_metrics['all_labels'],
    #     all_preds=cnn_metrics['all_preds'],
    #     target_layer_idx=9,             # Adjust based on your model's architecture
    #     num_images=30                    # Number of misclassified images to visualize
    # )
    # visualize_grad_cam(
    #     model=cnn_model,                # Choose which model to visualize ('resnet101_model' or 'cnn_model')
    #     test_paths=test_paths,
    #     all_labels=cnn_metrics['all_labels'],
    #     all_preds=cnn_metrics['all_preds'],
    #     target_layer_idx=9,             # Adjust based on your model's architecture
    #     num_images=5                    # Number of misclassified images to visualize
    # )


if __name__ == "__main__":
    main()

