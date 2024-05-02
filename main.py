import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

url = "https://www.openml.org/data/download/22102255/dataset"

r = requests.get(url, allow_redirects= True)

with open("dataset.txt", "wb") as f:
    f.write(r.content)

data = []
with open("dataset.txt", "r") as f:
     for line in f.read().split('\n'):
          if line.startswith("@") or line.startswith("%") or line =="":
               continue
          data.append(line)


col = []
with open("dataset.txt", "r") as f:  
     for line in f.read().split("\n"):
          if line.startswith("@ATTRIBUTE"):
               col.append(line.split(" ")[1])
# print(col)

with open("df.csv", "w") as f:
     f.write(",".join(col))
     f.write("\n")
     f.write("\n".join(data))

df = pd.read_csv("df.csv")
df.columns = col

#Categorize to binary
df['t_win'] = df.round_winner.astype('category').cat.codes

# Select only numeric columns for correlation calculation
numeric_cols = df.select_dtypes(include='number').columns
correlations = df[numeric_cols].corr()

# Print correlations with t_win
print(correlations["t_win"].apply(abs).sort_values(ascending=False).iloc[:25])
