import pandas as pd
from src.prepare_data import engineer, FORBIDDEN
def test_dataset_and_leakage():
    df=pd.read_csv("data/raw/ai4i2020.csv")
    assert len(df)==10000 and int(df["Machine failure"].sum())==339
    assert FORBIDDEN.isdisjoint(engineer(df).columns)
