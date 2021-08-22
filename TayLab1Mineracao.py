#!/usr/bin/env python
# coding: utf-8

# # Lab 01 - Mineração de Dados: Impactos da Representação

# Definição do objetivo
# 
# 1) Gerar diferentes vetores de características variando os valores de X e Y. 
# 
# 2) Utilizar um kNN (k=3 e distância Euclidiana), encontrar o conjunto de características que produziu os piores e melhores resultados de classificação. 
# 
# 3) Dividir a  base de dados em 50% para treinamento e 50% para validação. 
# 
# 3) Comparar as matrizes de confusão nesses dois casos
# 
# 4) Reportar quais foram as confusões resolvidas pela melhor representação.
# 
# 5) Verificar se é possível melhorar o resultados mudando os valores de k e métrica de distância.
# 

# In[ ]:


#Instalando pacotes
pip install -U scikit-learn
pip install progressbar2
pip install opencv-python


# In[1]:


#Carregando os arquivos
file='D:/_USER Tayana/Desktop/DSBD2021/digits/files.txt'
path='D:/_USER Tayana/Desktop/DSBD2021/digits'


# In[2]:


#Importando os pacotes
import os
import pandas as pd
import cv2
import progressbar


# In[17]:


w = 35
h = 35

X = []
y = []

listOfImages = []
  
arq = open(file)
lines = arq.readlines()


for line in  progressbar.progressbar(lines):
  aux = line.split('/')[1]
  image_name = aux.split(' ')[0]
  label = line.split(' ')[1]
  label = label.split('\n')
  
  f = path + '/data/' + image_name
  #print (f)
  
  ## opens the image using a single chanel (grey level)
  imagem = cv2.imread(f,0)
  
  listOfImages.append(f)
  
  ## rezise the image according to the parameters
  imagem = cv2.resize(imagem,(h,w))  #### <=====

  ## pixel count  
  v = []
  for i in range(w):
    for j in range(h):
      if(imagem[i][j] > 128):
        v.append(0)
      else:
        v.append(1)
  
  X.append(v)
  y.append(int(label[0]))


# In[18]:


## Save data
            
# Saving the extracted features (handcrafted) in a csv file
print ('Saving data...')
df = pd.DataFrame(X)
df.to_csv('D:/_USER Tayana/Desktop/DSBD2021/digits/X.csv', header=False, index=False)

# Saving the classes in a csv file
df_class = pd.DataFrame(y)
df_class.to_csv('D:/_USER Tayana/Desktop/DSBD2021/digits/y.csv', header=False, index=False)

print ('Done!')


# In[19]:


import pandas as pd
import numpy as np
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# In[20]:


# Labels
y = pd.read_csv('D:/_USER Tayana/Desktop/DSBD2021/digits/y.csv', header=None)
y=y.to_numpy()
y=np.ravel(y)
print(y.shape)


# In[21]:


# Features
X = pd.read_csv('D:/_USER Tayana/Desktop/DSBD2021/digits/X.csv', header=None)
X=X.to_numpy()
print(X.shape)


# In[22]:


#splits data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=30)


# In[27]:


#Criar um KNN
clfa = KNeighborsClassifier(n_neighbors=1, metric='euclidean')


# In[28]:


#clfa = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
print('Fitting KNN')
clfa = clfa.fit(X_train, y_train)


# In[29]:


# Predição do classificador
predicted=clfa.predict(X_test)
score=clfa.score(X_test, y_test)

matrix = confusion_matrix(y_test, predicted)


# In[30]:


# Print resultado
print("Accuracy = %.2f " % score)
print("Confusion Matrix:")
print(matrix)

print(classification_report(y_test, predicted, labels=[0,1, 2, 3,4,5,6,7,8,9]))


# # Lab 02 - Classificadores

# In[ ]:




