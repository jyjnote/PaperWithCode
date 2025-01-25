
def main():
    from data import load_data
    DATA_PATH=r"C:\PapersWithCode\04_NLP\data\review"
    train, test=load_data(DATA_PATH)

    train_arr = train["review"].to_numpy()
    test_arr = test["review"].to_numpy()
    target = train["target"].to_numpy().reshape(-1,1)

    model_name = "JKKANG/ALBERT-kor-emotion"

    from transformers import AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    from data import CollateFN
    collate_fn = CollateFN(tokenizer)

    from train import start
    start(train_arr,target,collate_fn)

if __name__ == "__main__":
    main()