# Rubén Sánchez Mayén A01378379
# 11/09/2022
# Random forests
# Datos: Automobile.csv
# Algoritmo para determinar la marca de un automóbil según algunas de sus especificaciones.

import pandas as pd
import random
import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Cambiar al directorio del script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Leer datos
df = pd.read_csv("Automobile_edited.csv")

# Codificar variables categoricas
# One-hot encoding
dfCat = df.select_dtypes(include="object")
dfCat = dfCat.drop('make',axis=1)
df = df.drop(dfCat.columns,axis=1)
label_encoder = LabelEncoder()
dfCat = pd.get_dummies(dfCat)
for i in range(len(dfCat.columns)):
  name = dfCat.columns[i]
  df[name] = dfCat[name]
print('Tamaño del dataset',df.shape)

# Dividir datos
X = df.drop('make',axis=1)
y = df['make']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.15)

# Generar Random forest
rnd_clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=30, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

# Calcular precision
y_pred_rf= rnd_clf.predict(X_test)
print("random forest", accuracy_score(y_test, y_pred_rf))

# Hacer prediccion, se toma una fila aleatoria del dataset
print('classes:', ' '.join(y.unique()))
pos = random.randint(0,len(X)-1)
print('-- Prediction --')
print('Expected: ',y[pos])
pred =  rnd_clf.predict(X.iloc[[pos]])
print("prediction:",pred[0])

# Variando el numero de arboles
acc_score = []
acc_range = range(1,101)
for i in acc_range:
  rnd_clf = RandomForestClassifier(n_estimators=i, max_leaf_nodes=30, n_jobs=-1, random_state=42)
  rnd_clf.fit(X_train, y_train)
  # Calcular precision
  y_pred_rf= rnd_clf.predict(X_test)
  acc_score.append(accuracy_score(y_test, y_pred_rf))

plt.plot(acc_range, acc_score)
plt.xlabel('Número de arboles')
plt.ylabel('Accuracy')
plt.xticks(range(0,len(acc_range),10))
plt.grid(axis='both',linestyle='--')
plt.show()

