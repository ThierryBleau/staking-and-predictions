import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

train_data = pd.read_csv('numerai_training_data.csv')
tour_data = pd.read_csv('numerai_tournament_data.csv')

train_bernie = train_data.drop([
        'id', 'data_type', 'target_charles', 'target_elizabeth',
        'target_jordan', 'target_ken', 'target_frank', 'target_hillary'],axis=1)

features = [f for f in list(train_bernie) if "feature" in f]

X_train = train_bernie[features]
Y_train = train_bernie['target_bernie']

model1 = linear_model.LogisticRegression()
model2 = RandomForestClassifier(n_estimators = 90)
#model3 = LinearSVC(penalty = 'l2')
model4 = KNeighborsClassifier()
model5 = MultinomialNB()
voting = VotingClassifier(estimators=[('LRC',model1),('RFC',model2),('KNN',model4),('MNB',model5)], voting='soft')
start = time.time()
print('Fitting the data...')
voting.fit(X_train,Y_train)

validation = tour_data[tour_data['data_type'] == 'validation']
X_test = validation[features]
Y_test = validation['target_bernie']
y_pred = voting.predict(X_test)

accuracy = accuracy_score(Y_test,y_pred)
print('accuracy: ', accuracy)
#print('Random gap: ', accuracy - random_rate)
print('roc_auc: ', roc_auc_score(Y_test,y_pred))
print('precision: ', precision_score(Y_test,y_pred))
print('recall: ',recall_score(Y_test,y_pred) )

cm = confusion_matrix(Y_test,y_pred)