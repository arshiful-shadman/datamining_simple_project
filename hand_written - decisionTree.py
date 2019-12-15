#%%
#All libraries required

# __ __ __ __ __ __ __
#|                    |
#|                    |
#|                    |
#|    HAND WRITTEN    | 28
#|        DIGIT       |
#|                    |
#|__ __ __ __ __ __ __|  
#          28

#28*28=784 pixels
from datetime import datetime
startTime = datetime.now()

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree




data=pd.read_csv("imageData.csv").as_matrix()

clf=DecisionTreeClassifier()

#training_dataset
xtrain=data[0:21000,1:]
train_label=data[0:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest=data[21000:,1:]
actual_label=data[21000:,0]

#d=xtest[15]# this is a 1
#d=xtest[16]# this is a 9
#d=xtest[17]# this is a 5
d=xtest[18]# this is a 2


d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
#print(clf.predict([xtest[15]])) # this is a 1
#print(clf.predict([xtest[16]])) # this is a 9
#print(clf.predict([xtest[17]])) # this is a 5
print(clf.predict([xtest[18]])) # this is a 2
pt.show()


p=clf.predict(xtest)
count=0
for i in range (0,21000):
    count+=1 if p[i]==actual_label[i] else 0
print("Decision Tree Accuracy=", (count/21000)*100)

print(confusion_matrix(actual_label,p))
print(classification_report(actual_label,p))
tree.plot_tree(clf)

print(datetime.now() - startTime)