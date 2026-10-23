import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split(path="data/creditcard.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
