
def main():
    from data import load_data
    DATA_PATH=r"C:\PapersWithCode\04_NLP\data\review"
    train, test=load_data(DATA_PATH)

    train_arr = train["review"].to_numpy()
    test_arr = test["review"].to_numpy()
    target = train["target"].to_numpy().reshape(-1,1)

    model_name = "ehdwns1516/klue-roberta-base-kornli"

    from model import load_model
    model=load_model(model_name)

    from train import start_training
    import torch
    device="cuda" if torch.cuda.is_available() else "cpu"
    start_training(model,train_arr,target,device)

if __name__ == "__main__":
    main()