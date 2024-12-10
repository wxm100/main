import pandas as pd

# Load Data
input_file = "soc-sign-bitcoinalpha.csv"
output_file = "simplified-soc-sign-bitcoinalpha.csv"

data = pd.read_csv(input_file, header=None, names=["SOURCE", "TARGET", "RATING", "TIME"])

# Simple RATING
data["RATING"] = data["RATING"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Delete TIME
data = data.drop(columns=["TIME"])

# New CSV file
data.to_csv(output_file, index=False, header=False)

print(f"Simple {output_file}")
