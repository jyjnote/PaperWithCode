



def down_and_load():
    import os
    from torchvision import datasets
    import torchvision.transforms as transforms
    # specift the data path
    path2data = './data'

    # if not exists the path, make the directory
    if not os.path.exists(path2data):
        os.mkdir(path2data)

    # load dataset
    train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
    val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())

    # define image transformation
    transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(299)
    ])

    train_ds.transform = transformation
    val_ds.transform = transformation
    return train_ds, val_ds