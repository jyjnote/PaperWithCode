from torchvision import transforms, datasets # transforms : 데이터를 조작하고 학습에 적합하게 만듦.
from torch.utils.data import Dataset, DataLoader
# Transforms to resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize(227),  # Resize to fit AlexNet input
    transforms.ToTensor()])  # Convert to tensor

# FashionMNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

validation_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)