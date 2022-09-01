# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:25:40 2020

@author: monte
"""

#CARGAMOS EL DATASET
from sklearn.model_selection import train_test_split
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing


#CARGAMOS EL DATASET
df = pd.read_csv('./teleCust1000t.csv')
df.head()

df['custcat'].value_counts()

#VIZUALIZAMOS EL HISTOGRAMA D
#df.hist(column='income', bins=50)

#MUESTRA LOSTITULOS DE LAS COLUMNAS
df.columns

#Para utilizar la librería scikit-learn, tenemos que convertir el data frame de Panda en un Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

#AQUI SE MUESTRAN LAS ETIQUETAS DE CADA TIPO DE MASCOTA SON 4 EN TOTAL
y = df['custcat'].values
y[0:5]

#NORMALIZACION DE LOS DATOS
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#ENTRENAMIENTO DEL MODELO
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Set de Entrenamiento:', X_train.shape,  y_train.shape)
print ('Set de Prueba:', X_test.shape,  y_test.shape)



#METODO DE LOS K-VECINOS MAS PROXIMOS
#ENTRENAMIENCO CON K=4
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

k = 4
#Entrenar el Modelo y Predecir  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#PREDECIMOS EL SET DE PRUEBA
yhat = neigh.predict(X_test)
yhat[0:5]

#EVALUACION DE LA CERTEZA
print("Entrenar el set de Certeza K=4: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Probar el set de Certeza k=4: ", metrics.accuracy_score(y_test, yhat))

####################################################################

# AUMENTAMOS EL VALOR DE K Y VEMOS DONDE ESTA LA MAYOR PRECISION.
Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Entrenar el Modelo y Predecir  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    print(mean_acc[n-1])
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

#Grafica de la certeza
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Certeza ', '+/- 3xstd'))
plt.ylabel('Certeza ')
plt.xlabel('Número de Vecinos (K)')
plt.tight_layout()
plt.show()

print( "La mejor aproximación de certeza fue con ", mean_acc.max(), "con k=", mean_acc.argmax()+1) 
