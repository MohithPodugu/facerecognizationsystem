import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import json
import os

# Define transformations with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "image_dataset"
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

# Count dataset sizes
dataset_info = {
    "train_samples": len(train_dataset),
    "test_samples": len(test_dataset),
    "total_samples": len(train_dataset) + len(test_dataset)
}

with open("dataset_info.json", "w") as f:
    json.dump(dataset_info, f)

# Save class names
class_names_dict = {str(i): name for i, name in enumerate(train_dataset.classes)}
with open("class_names.json", "w") as f:
    json.dump(class_names_dict, f)

# Load pre-trained VGG19 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19(weights="IMAGENET1K_V1")
model.classifier[6] = nn.Linear(4096, len(train_dataset.classes))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in DataLoader(train_dataset, batch_size=32, shuffle=True):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataset):.4f}, Accuracy: {accuracy:.2f}%")

# Save trained model
torch.save(model.state_dict(), "vgg19_face_recognition.pth")
print("Training complete. Model saved.")
