import pandas as pd
from sklearn.model_selection import train_test_split

class DataPrepration:

    def __init__(self) -> None:
        
        # self.file_path = "..\data\housing_data.csv"
        self.file_path = "/app/data/housing_data.csv"

    def prepare_data(self):
        df = pd.read_csv(self.file_path)

        # X = df.drop(columns=["Price"])
        X = df[["HouseAge"]]
        y = df["Price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        return X_train, X_test, y_train, y_test

if __name__=="__main__":
    pass
    