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
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import argparse

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Define model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 50

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

class CustomImageDataset(Dataset):
    def __init__(self, data_folder, data_file, transform=None, exclude_yaw=None):
        self.data_folder = data_folder
        self.data_file = data_file
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data(exclude_yaw)

    def load_data(self, exclude_yaw=None):
        for idx, data_file in enumerate(self.data_file):
            full_path = os.path.join(self.data_folder, data_file)
            data = np.load(full_path)
            images = data["images"]
            len_ori = images.shape[0]
            labels = np.full(images.shape[0], idx)
            if exclude_yaw is not None:
                poses = data["poses"]
                xyzrpy = extrinsic_matrix_to_xyzrpy(poses)
                yaw = xyzrpy[5]
                for interval in exclude_yaw:
                    mask = ~np.logical_and(yaw > interval[0], yaw < interval[1])
                    images = images[mask]
                    labels = labels[mask]
                    yaw = yaw[mask]
            print(f"Trained yaw portion: {images.shape[0] / len_ori}")
            
            images_resized = []
            for img in images:
                img = img * 255
                img_resized = Image.fromarray(img.astype(np.uint8))
                img_resized = img_resized.resize((IMAGE_SIZE, IMAGE_SIZE))
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

def extrinsic_matrix_to_xyzrpy(T):
    x, y, z = T[:, 0, 3], T[:, 1, 3], T[:, 2, 3]
    R = T[:, :3, :3]

    def rotation_matrix_to_rpy(R):
        pitch = -np.arcsin(R[:, 2, 0])
        roll = np.zeros_like(pitch)
        yaw = np.arctan2(-R[:, 0, 1], R[:, 1, 1])
        mask = np.abs(np.cos(pitch)) > np.finfo(float).eps
        roll[mask] = np.arctan2(R[:, 2, 1], R[:, 2, 2])
        yaw[mask] = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        return roll, pitch, yaw

    roll, pitch, yaw = rotation_matrix_to_rpy(R)
    return np.array([x, y, z, roll, pitch, yaw])

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
        # self.layer3 = BasicBlock(16, 32, stride=2, bn=True, kernel=3)
        self.fc = nn.Linear(2704, 5)  # Output 5 classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        return x


def pgd_attack(model, images, labels, criterion, epsilon, alpha, num_iter, device):
    """
    Performs a PGD attack on a batch of images.
    
    Args:
        model: The neural network.
        images: Original input images.
        labels: True labels corresponding to the images.
        criterion: Loss function.
        epsilon: Maximum perturbation.
        alpha: Step size for each iteration.
        num_iter: Number of iterations.
        device: Device (cpu or cuda).
    
    Returns:
        adv_images: Adversarial examples.
    """
    # Starting at a randomly perturbed point within the epsilon ball
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1).to(device)

    for _ in range(num_iter):
        # Enable gradient computation for adversarial images
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        
        # Zero all existing gradients
        model.zero_grad()
        if adv_images.grad is not None:
            adv_images.grad.data.zero_()
        
        # Backward pass
        loss.backward()
        
        # Get the gradient sign on the adversarial images
        grad_sign = adv_images.grad.data.sign()
        
        # Take a step in the direction of the gradient sign
        adv_images = adv_images + alpha * grad_sign
        
        # Project the perturbation: ensure it's within the epsilon ball of the original images
        perturbation = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = images + perturbation
        
        # Clip to the valid image range (e.g., [0,1])
        adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    return adv_images


def adversarial_train_model(model, 
                            train_loader, 
                            test_loader, 
                            criterion, 
                            optimizer, 
                            weights_path,
                            num_epochs=100,
                            epsilon=0.03,    # maximum perturbation
                            alpha=0.007,     # step size for each PGD iteration
                            num_iter=10):    # number of PGD iterations
    """
    Adversarial training loop using a PGD attack.
    Trains on both clean and adversarial images in each batch.
    """
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # -------------------------------------
            # Generate adversarial examples using PGD
            # -------------------------------------
            adv_images = pgd_attack(model, images, labels, criterion, epsilon, alpha, num_iter, device)

            # -------------------------------------
            # Combine clean and adversarial images
            # -------------------------------------
            combined_images = torch.cat([images, adv_images], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()
            outputs = model(combined_images)
            loss = criterion(outputs, combined_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy on the clean images only
            with torch.no_grad():
                outputs_clean = model(images)
                _, predicted_clean = torch.max(outputs_clean.data, 1)
                total += labels.size(0)
                correct += (predicted_clean == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Evaluate on the test set every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_accuracy = evaluate_model(model, test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%")

    # Save the trained model weights
    torch.save(model.state_dict(), weights_path)
    print('Model saved successfully!')


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

    input_bounds = np.load('verification/tinydozer_error_0005_method_crown_features_10_4_128_2_20000_samples_32_inputdim_6_xdown_1_ydown_1.npz')
    lower = prepare_input_bounds(input_bounds['image_lb']).to(device)
    upper = prepare_input_bounds(input_bounds['image_ub']).to(device)
    image_noptb = prepare_input_bounds(input_bounds['image_noptb']).to(device)

    cex_x, cex_y = parse_cex(f'../tinydozer_{IMAGE_SIZE}_0005.cex')
    image_cex = torch.Tensor(cex_x).to(lower)
    image_cex = image_cex.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    fig, axes = plt.subplots(1, 4)
    image_toshow = image_noptb[0].permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(image_toshow)
    axes[0].set_title("Noptb")
    image_toshow = lower[0].permute(1, 2, 0).cpu().numpy()
    axes[1].imshow(image_toshow)
    axes[1].set_title("Lower")
    image_toshow = upper[0].permute(1, 2, 0).cpu().numpy()
    axes[2].imshow(image_toshow)
    axes[2].set_title("Upper")
    image_toshow = image_cex[0].permute(1, 2, 0).cpu().numpy()
    axes[3].imshow(image_toshow)
    axes[3].set_title("Cex")
    plt.show()
    plt.savefig(f'tinydozer_{IMAGE_SIZE}_0005.png')

    # Select the image at the specified index
    testimg = images[image_idx]
    testimg = cv2.resize(testimg, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to 64x64  
    testimg = torch.Tensor(testimg).to(device)
    testimg = testimg.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)

    # Run prediction
    with torch.no_grad():
        outputs = model(testimg)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted class: {categories[predicted.item()]}")


def create_model():
    model = ImageClassificationModel()
    return model


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
    testimg = cv2.resize(testimg, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to 64x64  
    testimg = torch.Tensor(testimg).to(device)
    testimg = testimg.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)

    eps = 0.005
    input_bounds = {'lower': testimg - eps, 'upper': testimg + eps}
    torch.save(input_bounds, 'verification/input_bounds.pth')

    model = BoundedModule(model, testimg, bound_opts={"optimize_bound_args": {"iteration": 60, 'lr_alpha': 5}})
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps)


    my_input = BoundedTensor(testimg, ptb)
    prediction = model(my_input)
    lb, ub = model.compute_bounds(x=(my_input,), method="crown")
    print('lb:',lb)
    print('ub:',ub)

    # input_ptb=testimg+(2*torch.rand(100,3,16,16,device=device)-1)*eps
    # prediction=model(input_ptb)
    
    # print(prediction)

    # Run prediction
    # with torch.no_grad():
    #     outputs = model(testimg)
    #     _, predicted = torch.max(outputs, 1)
    #     print(f"Predicted class: {categories[predicted.item()]}")


def prepare_input_bounds(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = torch.Tensor(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def parse_cex(cex_file):
    """Parse the saved counter example file."""
    x_dict = defaultdict(int)
    y_dict = defaultdict(int)
    max_x_dim = -1
    max_y_dim = -1
    reg_match = re.compile(r"\(+\s?([XY])_(\d+)\s+([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\s?\)+")
    with open(cex_file) as f:
        for line in f:
            m = reg_match.match(line.strip())
            if m:
                xy, dim, val = m.group(1), m.group(2), m.group(3)
                dim = int(dim)
                val = float(val)
                if xy == "X":
                    max_x_dim = max(max_x_dim, dim)
                    x_dict[dim] = val
                elif xy == "Y":
                    max_y_dim = max(max_y_dim, dim)
                    y_dict[dim] = val
    max_x_dim += 1
    max_y_dim += 1
    print(f"Loaded input variables with dimension {max_x_dim} with {len(x_dict)} nonzeros.")
    print(f"Loaded output variables with dimension {max_y_dim} with {len(y_dict)} nonzeros.")
    x = np.zeros(max_x_dim)
    y = np.zeros(max_y_dim)
    for i in range(max_x_dim):
        x[i] = x_dict[i]
    for i in range(max_y_dim):
        y[i] = y_dict[i]
    return x, y

def prepare_input_bounds(image):
    # image: (H, W, C) or (N, H, W, C)
    # If image is in batch, assume it doesn't need to be resized.
    if image.ndim == 4:
        image = torch.Tensor(image)
        image = image.permute(0, 3, 1, 2)
    else:
        image = cv2.resize(image, (50, 50))
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
    return image

def predict_images(images, classidx, weights_path):
    model = ImageClassificationModel().to(device)
    model.load_state_dict(torch.load(weights_path))  # Load model weights
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    _, results = outputs.max(dim=1)
    acc = torch.sum(results == classidx) / images.shape[0]
    print(f"acc: {acc}")
    return results
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("choice", type=str)
    args = parser.parse_args()
    choice = args.choice

    epsilon=0.03    # maximum perturbation
    alpha=0.007     # step size for each PGD iteration
    num_iter=10     # number of PGD iterations

    exclude_yaw = None

    # Path to data folder and data
    data_folder = './data/'
    data_file = ['chair_data.npz', 'lego_data.npz', 'ficus_data.npz', 'hotdog_data.npz', 'mic_data.npz']
    categories = ['chair', 'lego', 'ficus', 'hotdog', 'mic']
    weight_folder = './weights/'
    weights_filename = f'model_layer12_weights_{IMAGE_SIZE}.pth'
    if choice == 'advtrain' or choice == 'predict_images':
        weights_filename = f'model_layer12_weights_advtrain_{IMAGE_SIZE}_eps{epsilon}_alpha{alpha}_iter{num_iter}.pth'
    if exclude_yaw is not None:
        weights_filename = weights_filename[:-4] + f'_exclude_yaw.pth'
    weights_path = os.path.join(weight_folder, weights_filename)
    print(f"weights_path: {weights_path}")

    model = ImageClassificationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs=50

    if choice == 'train':
        # Create dataset and dataloader
        dataset = CustomImageDataset(data_folder, data_file, transform=transform, exclude_yaw=exclude_yaw)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train the model
        train_model(model, train_loader, test_loader, criterion, optimizer, weights_path, num_epochs)
    
    elif choice == 'advtrain':
        # Create dataset and dataloader
        dataset = CustomImageDataset(data_folder, data_file, transform=transform, exclude_yaw=exclude_yaw)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train the model with adversarial training
        adversarial_train_model(model, train_loader, test_loader, criterion, optimizer, weights_path, num_epochs)

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
    
    elif choice == 'predict_images':
        inputs = np.load('verification/predicted_images_10000.npz')
        images = prepare_input_bounds(inputs["images"]).to(device)
        poses = inputs["poses"]
        classidx = 1
        results = predict_images(images, classidx, weights_path).cpu()
        xyzrpy = extrinsic_matrix_to_xyzrpy(poses)
        all_yaw = xyzrpy[5]
        mask_correct = results == classidx
        yaw_correct = all_yaw[mask_correct]
        yaw_incorrect = all_yaw[~mask_correct]
        
        fix, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        # fix, ax = plt.subplots()
        ax.scatter(yaw_correct, np.ones_like(yaw_correct), color='green', s=100, label='correct')
        ax.scatter(yaw_incorrect, np.ones_like(yaw_incorrect), color='red', s=100, label='incorrect')
        ax.set_thetamin(-180)
        ax.set_thetamax(180)
        ax.set_xticks(np.radians([-180, -90, 0, 90, 180]))
        ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
        ax.set_yticklabels([])
        ax.legend()
        plt.show()
        plt.savefig('rotation_classification.png')
