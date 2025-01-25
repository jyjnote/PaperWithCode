
def main():
    import warnings,torch
    warnings.filterwarnings("ignore")
    from data import data_load,data_processing
    warnings.filterwarnings("ignore", message=".*contains `beta`.*")
    warnings.filterwarnings("ignore", message=".*contains `gamma`.*")
    import numpy as np
    
    DATA_PATH=r"C:\PapersWithCode\04_NLP\data\imdb\imdb_700.csv"

    data,target=data_load(DATA_PATH)
    train_data=data_processing(data)

    train_input_ids = np.array(train_data['input_ids'])
    train_token_type_ids = np.array(train_data['token_type_ids'])
    train_attention_mask = np.array(train_data['attention_mask'])

    device="cuda" if torch.cuda.is_available() else "cpu"

    from train import start_training
    
    start_training(train_token_type_ids, 
                   train_token_type_ids,
                   train_attention_mask,
                   target,
                   device)

if __name__ == "__main__":
    main()