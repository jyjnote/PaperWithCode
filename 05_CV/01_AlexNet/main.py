
def main():
    from data import transform,training_data,validation_data
    from torch.utils.data import Dataset, DataLoader
    import torch

    # training_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    
    # validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)


    from model import Alexnet
    model=Alexnet()


    from train import start_train
    start_train(training_data,validation_data,model)

if __name__ == "__main__":
    main()