from mrmr import mrmr_classif
from sklearn.datasets import make_classification
import pandas as pd
csv = "ObesityDataSet_raw_and_data_sinthetic.csv.disc"

df = pd.read_csv(f"~/datasets/{csv}",sep=",")
# use mrmr classification
selected_features = mrmr_classif(df, df.columns, K = 10)