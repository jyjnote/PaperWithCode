import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_and_process_data():

    # 다른 경로에서 파일 불러오기
    from glob import glob
    csv_files = glob("C:\\PapersWithCode\\03_Deep_Learning\\data\\titanic_*.csv")
    (test, train) = csv_files
    train = pd.read_csv(train)
    test = pd.read_csv(test)

    # 결측값 처리
    for col in train.columns[train.isnull().sum() > 0]:
        if pd.api.types.is_numeric_dtype(train[col]):
            train[col] = train[col].fillna(train[col].mean())
        else:
            train[col] = train[col].fillna(train[col].mode()[0])
    
    for col in test.columns[test.isnull().sum() > 0]:
        if pd.api.types.is_numeric_dtype(test[col]):
            test[col] = test[col].fillna(test[col].mean())
        else:
            test[col] = test[col].fillna(test[col].mode()[0])

    # 특성 선택
    cols = ["age", "sibsp", "parch", "fare", "pclass", "gender", "embarked"]
    train_ft = train[cols].copy()
    test_ft = test[cols].copy()

    # 원-핫 인코딩
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_ft[['gender', 'embarked']])
    tmp_train = pd.DataFrame(enc.transform(train_ft[['gender', 'embarked']]).toarray(), columns=enc.get_feature_names_out())
    tmp_test = pd.DataFrame(enc.transform(test_ft[['gender', 'embarked']]).toarray(), columns=enc.get_feature_names_out())
    train_ft = pd.concat([train_ft, tmp_train], axis=1)
    test_ft = pd.concat([test_ft, tmp_test], axis=1)

    # 불필요한 컬럼 제거
    train_ft = train_ft.drop(columns=['gender', 'embarked'])
    test_ft = test_ft.drop(columns=['gender', 'embarked'])

    # 스케일링
    scaler = MinMaxScaler()
    train_ft = scaler.fit_transform(train_ft)
    test_ft = scaler.transform(test_ft)

    # 타겟 변수
    target = train["survived"].to_numpy().reshape(-1,1)

    return train_ft, test_ft, target
