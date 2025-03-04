import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import cv2

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Define model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

class CustomImageDataset(Dataset):
    def __init__(self, data_folder, data_file, transform=None):
        self.data_folder = data_folder
        self.data_file = data_file
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for idx, data_file in enumerate(self.data_file):
            full_path = os.path.join(self.data_folder, data_file)
            data = np.load(full_path)
            images = data["images"]
            labels = np.full(images.shape[0], idx)

            images_resized = []
            for img in images:
                img = img * 255
                img_resized = Image.fromarray(img.astype(np.uint8))
                img_resized = img_resized.resize((16, 16))
                images_resized.append(np.array(img_resized))

            self.data.append(np.array(images_resized))
            self.labels.append(labels)

        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx].astype(np.uint8))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2, stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImageClassificationModel(nn.Module):
    def __init__(self):
        super(ImageClassificationModel, self).__init__()
        self.layer1 = BasicBlock(3, 8, stride=2, bn=True, kernel=3)
        self.layer2 = BasicBlock(8, 16, stride=2, bn=True, kernel=3)
        self.layer3 = BasicBlock(16, 32, stride=2, bn=True, kernel=3)
        self.fc = nn.Linear(32* 1 * 4, 5)  # Output 5 classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        return x

def train_model(model, train_loader, test_loader, criterion, optimizer, weights_path, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Evaluate on the test set every 50 epochs
        if (epoch + 1) % 10 == 0:
            test_accuracy = evaluate_model(model, test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%")

    # Save the trained model weights
    torch.save(model.state_dict(), weights_path)
    print('Model saved successfully!')

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def predict(data_folder, dataname, image_idx, weights_path):
    model = ImageClassificationModel().to(device)
    model.load_state_dict(torch.load(weights_path))  # Load model weights
    model.eval()

    datapath = os.path.join(data_folder, dataname + '_data.npz')

    # Load the image data from .npz
    data = np.load(datapath)
    images = data["images"]

    # Select the image at the specified index
    testimg = images[image_idx]
    testimg = cv2.resize(testimg, (16, 16))  # Resize to 64x64  
    testimg = torch.Tensor(testimg).to(device)
    testimg = testimg.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)

    # Run prediction
    with torch.no_grad():
        outputs = model(testimg)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted class: {categories[predicted.item()]}")

def bound(data_folder, dataname, image_idx, weights_path):
    model = ImageClassificationModel().to(device)
    model.load_state_dict(torch.load(weights_path))  # Load model weights
    model.eval()

    datapath = os.path.join(data_folder, dataname + '_data.npz')

    # Load the image data from .npz
    data = np.load(datapath)
    images = data["images"]

    # Select the image at the specified index
    testimg = images[image_idx]
    testimg = cv2.resize(testimg, (16, 16))  # Resize to 64x64  
    testimg = torch.Tensor(testimg).to(device)
    testimg = testimg.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)

    model = BoundedModule(model, testimg)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.01)


    my_input = BoundedTensor(testimg, ptb)
    prediction = model(my_input)
    lb, ub = model.compute_bounds(x=(my_input,), method="alpha-crown")
    print('lb:',lb)
    print('ub:',ub)

    input_ptb=testimg+(2*torch.rand(100,3,16,16,device=device)-1)*0.01
    prediction=model(input_ptb)
    
    print(prediction)

    # Run prediction
    # with torch.no_grad():
    #     outputs = model(testimg)
    #     _, predicted = torch.max(outputs, 1)
    #     print(f"Predicted class: {categories[predicted.item()]}")

if __name__ == '__main__':
    choice = 'bound'

    # Path to data folder and data
    data_folder = './data/'
    data_file = ['chair_data.npz', 'lego_data.npz', 'ficus_data.npz', 'hotdog_data.npz', 'mic_data.npz']
    categories = ['chair', 'lego', 'ficus', 'hotdog', 'mic']
    weight_folder = './weights/'
    weights_filename = 'model_weights.pth'
    weights_path = os.path.join(weight_folder, weights_filename)

    model = ImageClassificationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=50

    if choice == 'train':
        # Create dataset and dataloader
        dataset = CustomImageDataset(data_folder, data_file, transform=transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train the model
        train_model(model, train_loader, test_loader, criterion, optimizer, weights_path, num_epochs)

    elif choice == 'predict':
        data_folder = './data/'  # Specify the folder where data is stored
        dataname = 'chair'  # Use one of ['chair', 'lego', 'ficus', 'hotdog', 'mic']
        image_idx = 0  # Specify the image index to predict (e.g., 0)
        predict(data_folder, dataname, image_idx, weights_path)

    elif choice == 'bound':
        data_folder = './data/'  # Specify the folder where data is stored
        dataname = 'chair'  # Use one of ['chair', 'lego', 'ficus', 'hotdog', 'mic']
        image_idx = 0  # Specify the image index to predict (e.g., 0)
        bound(data_folder, dataname, image_idx, weights_path)
