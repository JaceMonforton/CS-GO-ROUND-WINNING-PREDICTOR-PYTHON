import requests

url = "https://www.openml.org/data/download/22102255/dataset"

r = requests.get(url, allow_redirects= True)

with open("dataset.txt", "wb") as f:
    f.write(r.content)
    