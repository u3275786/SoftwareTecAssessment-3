import pandas as pd


file_path = '/Users/joelvanheel/Downloads/NFLX.csv.xls'

# Read the dataset into a pandas DataFrame
df = pd.read_csv(file_path)
df_cleaned = df.drop_duplicates()

print(df_cleaned.head())