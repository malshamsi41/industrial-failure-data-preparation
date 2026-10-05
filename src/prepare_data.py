from pathlib import Path
import pandas as pd

BASE = ["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
FORBIDDEN = {"Machine failure","TWF","HDF","PWF","OSF","RNF","UDI","Product ID"}

def engineer(df):
    x = df[BASE].copy()
    x["Temperature difference [K]"] = x["Process temperature [K]"] - x["Air temperature [K]"]
    x["Power proxy"] = x["Torque [Nm]"] * x["Rotational speed [rpm]"]
    x["Torque x wear"] = x["Torque [Nm]"] * x["Tool wear [min]"]
    return x

def audit(df):
    return pd.DataFrame({
        "rows":[len(df)], "columns":[df.shape[1]], "missing_values":[int(df.isna().sum().sum())],
        "failures":[int(df["Machine failure"].sum())], "failure_rate_pct":[100*df["Machine failure"].mean()]
    })

def main():
    root=Path(__file__).resolve().parents[1]
    df=pd.read_csv(root/"data/raw/ai4i2020.csv")
    out=root/"outputs/tables"; out.mkdir(parents=True,exist_ok=True)
    audit(df).to_csv(out/"data_audit.csv",index=False)
    df["Machine failure"].value_counts().rename_axis("class").reset_index(name="count").to_csv(out/"class_counts.csv",index=False)
    x=engineer(df)
    assert FORBIDDEN.isdisjoint(x.columns)
    pd.concat([x,df[["Machine failure"]]],axis=1).to_csv(out/"model_ready_data.csv",index=False)
    print(audit(df).to_string(index=False))
    print("Model-ready predictors:", list(x.columns))
if __name__=="__main__": main()
