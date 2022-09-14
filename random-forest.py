# Rubén Sánchez Mayén A01378379
# 13/09/2022
# Random forests
# Datos: Automobile.csv
# Algoritmo para determinar la marca de un automóbil según algunas de sus especificaciones.

import pandas as pd
import random
import os

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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

accuracy_arr = []
n_iteration = []

for i in range(15):
  # Dividir datos
  X = df.drop('make',axis=1)
  y = df['make']
  X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.15)

  # Generar Random forest
  rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=25, n_jobs=-1, random_state=42, max_depth=15)
  rnd_clf.fit(X_train, y_train)

  # Calcular precision
  y_pred= rnd_clf.predict(X_test)
  acc_score= accuracy_score(y_test, y_pred)
  print('random forest',acc_score)
  accuracy_arr.append(acc_score)
  n_iteration.append(i+1)

plt.plot(n_iteration,accuracy_arr)
plt.ylim(0,1.2)
plt.title('Varianza')
plt.xlabel('Número de iteración')
plt.ylabel('Accuracy')
plt.xticks(n_iteration)
plt.grid(axis='both',linestyle='--',alpha=0.5)
plt.show()

# Matriz de confusion
cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
plt.figure(figsize = (10,10))
sn.heatmap(cm_df, annot = True)
plt.title('Nivel de Sesgo')
plt.show()

# Hacer prediccion, se toma una fila aleatoria del dataset
#print('classes:', ' '.join(y.unique()))
#pos = random.randint(0,len(X)-1)
#print('-- Prediction --')
#print('Expected: ',y[pos])
#pred =  rnd_clf.predict(X.iloc[[pos]])
#print("prediction:",pred[0])

# Variando el numero de arboles
#acc_score = []
#acc_range = range(1,21)
#for i in acc_range:
#  rnd_clf = RandomForestClassifier(n_estimators=i, max_leaf_nodes=30, n_jobs=-1, random_state=42)
#  rnd_clf.fit(X_train, y_train)
  # Calcular precision
#  y_pred_rf= rnd_clf.predict(X_test)
# acc_score.append(accuracy_score(y_test, y_pred_rf))

#plt.plot(acc_range, acc_score)
#plt.xlabel('Número de arboles')
#plt.ylabel('Accuracy')
#plt.xticks(range(0,len(acc_range),10))
#plt.grid(axis='both',linestyle='--')
#plt.show()

