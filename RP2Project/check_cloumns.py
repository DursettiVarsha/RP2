import pandas as pd

df = pd.read_csv("data/text/depression_text.csv", low_memory=False)

print(df.columns)
print(df.head())
