# torchvision 버전: 0.20.1
from torchvision import transforms
from torchvision import datasets

# train_transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     # 전처리 추가
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4638, 0.4585, 0.4297], std=[1.0, 1.0, 1.0])
# ])

# valid_transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.4638, 0.4585, 0.4297], std=[1.0, 1.0, 1.0])
# ])
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 like ImageNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])
#! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Download and load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root="data",  # Directory to store the dataset
    train=True,   # Train set
    download=True,
    transform=transform
)
#! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
valid_dataset = datasets.CIFAR10(
    root="data",
    train=False,  # Test set
    download=True,
    transform=transform
)
#! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# def load_train(path):
#     train_dataset = datasets.ImageFolder(path, train_transform)
#     return train_dataset

# def load_valid(path):
#     valid_dataset = datasets.ImageFolder(path, valid_transform)
#     return valid_dataset