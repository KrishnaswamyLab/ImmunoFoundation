import pandas as pd

af3_csv = "/home/am3826/scratch_pi_sk2433/am3826/iedb_af3/merged_af3.csv"
df = pd.read_csv(af3_csv)
display(df.head())
print(f"Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df['immunogenicity'].value_counts())
print(df.isnull().sum())
