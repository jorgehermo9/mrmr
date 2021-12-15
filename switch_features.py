#Utility to change the class feature column to first column on the dataset.

import pandas as pd
csv = "~/datasets/iris.data.disc"

df = pd.read_csv(csv,sep=",")

target_columns = ["class"]
columns = df.columns.tolist()

for index,column in enumerate(columns):
	if column in target_columns:
		target_index= index

aux = df.iloc[:,0]
aux_column = columns[0]
to_switch = df.iloc[:,target_index]
to_switch_column = columns[target_index]

df.iloc[:,0] = to_switch
df.iloc[:,target_index] = aux

columns[0] = to_switch_column
columns[target_index] =aux_column

new_df = pd.DataFrame(data=df.values,columns=columns)
new_df.to_csv(f"{csv}.peng",index=False)