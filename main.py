import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras

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
df['bomb_planted'] = df.bomb_planted.astype('category').cat.codes

# Select only numeric columns for correlation calculation
numeric_cols = df.select_dtypes(include='number').columns

correlations = df[numeric_cols].corr()

# Print correlations with t_win (top 25)
# print(correlations["t_win"].apply(abs).sort_values(ascending=False).iloc[:25])

selected_col = []

for col in col+["t_win"]:
   try:
       if abs(correlations[col]["t_win"]) > 0.15:
           selected_col.append(col)
   except KeyError:
        pass
df_selected = df[selected_col]
# print(df_selected)


# plt.figure(figsize=(18,12))
# sns.heatmap(df_selected.corr().sort_values(by="t_win"), annot=True, cmap="YlGnBu")
# plt.show()

# df_selected.hist(figsize=(18,12))
# plt.show()

# df_selected.info()

#Train - Test

X, y = df_selected.drop(['t_win'], axis=1), df_selected["t_win"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)



# param_grid = {
#      "n_neighbors": list(range(5, 17, 2)), 
#      "weights": ["uniform", "distance"]
# }

# knn = KNeighborsClassifier(n_jobs=4)
# knn.fit(x_train_scaled, y_train)

# clf = RandomizedSearchCV(knn, param_grid, n_jobs=4, n_iter=3, verbose=2, cv=3)
# clf.fit(x_train_scaled, y_train)
# knn = clf.best_estimator_

# score = knn.score(x_test_scaled, y_test)


# #Random forest classifier
# forest = RandomForestClassifier(n_jobs=4)
# # forest.fit(x_train_scaled, y_train)
# score = forest.score(x_test_scaled, y_test)
# print(score)

#tensorflow

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(20,)))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5)
x_train_scaled_train, x_valid, y_train_train, y_valid = train_test_split(x_train_scaled, y_train, test_size=0.15)
model.fit(x_train_scaled_train, y_train_train, epochs=3, callbacks=[early_stopping_cb], validation_data =(x_valid,y_valid))

print(model.evaluate(x_test_scaled, y_test))