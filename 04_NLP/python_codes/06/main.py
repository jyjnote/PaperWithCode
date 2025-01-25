from data_processing import data_processing,ReviewDataset
from train_test import start_training
def main():
    (train,target)=data_processing()
    start_training(train,target,ReviewDataset)

if __name__ == '__main__':
    main()