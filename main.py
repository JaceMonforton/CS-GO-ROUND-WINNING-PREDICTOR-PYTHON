import requests
import pandas as pd

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
print(df)

