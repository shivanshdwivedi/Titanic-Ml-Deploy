# For data analysis and data wragling
import pandas as pd
import numpy as np
import pickle

# For data visualization and graph plotting

# For Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('./train.csv')
dataset.head(4)
df = dataset[['Pclass','Sex', 'Embarked','Age','Fare','Survived']]
df.head()
df.isnull().sum()
df.fillna({'Age': df.Age.mean() ,
          'Embarked': 'S'} , inplace = True)
df.head()
df.isnull().sum()

X = df.drop(columns='Survived').values
Y = df['Survived'].values
print(X)
print(Y)


em = LabelEncoder()
X[:,1] = em.fit_transform(X[:,1])
X[:,2] = em.fit_transform(X[:,2])
print(X)

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)
print(X_train)

# =============================================================================
# from sklearn.preprocessing import StandardScaler
# scale = StandardScaler()
# X_train = scale.fit_transform(X_train)
# X_test = scale.fit_transform(X_test)
# print(X_train)
# 
# 
# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf' , degree= 3)
# print(classifier)
# classifier.fit(X_train , Y_train)
# 
# y_pred = classifier.predict(X_test)
# print(y_pred)
# 
# 
# =============================================================================

classifier = RandomForestClassifier(n_jobs =  -1,
    n_estimators= 500,
     warm_start= True,
    max_depth= 6,
    min_samples_leaf= 2,
    max_features = 'sqrt',
    verbose =  0)
classifier.fit(X_train , Y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test , y_pred)
print(score)

pickle.dump(classifier, open('model.pkl','wb'))
