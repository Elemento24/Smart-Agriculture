import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# import lazypredict
# from lazypredict.Supervised import LazyClassifier
import pickle



df = pd.read_csv('../datasets/Crop_recommendation.csv')
# df.sample(10)

target = ['label']
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, shuffle=True)

# def lazyPredict():
#     clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
#     models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# def GaussModel():
#     NaiveBayes = GaussianNB()
#     NaiveBayes.fit(X_train,y_train)
#     predicted_values = NaiveBayes.predict(X_test)
#     x = metrics.accuracy_score(y_test, predicted_values)
#     # print("Naive Bayes's Accuracy is: ", x)
#     # print(classification_report(y_test,predicted_values))

#     score = cross_val_score(NaiveBayes,X,y,cv=5)

#     NB_pkl_filename = 'NBClassifier.pkl'
#     NB_Model_pkl = open(NB_pkl_filename, 'wb')
#     pickle.dump(NaiveBayes, NB_Model_pkl)
#     NB_Model_pkl.close()



# GaussModel()

def predictedCropRecom(nitrogen, phosphorus, potash,
                      temp, humid, ph, rainfall):
    
    NaiveBayes = GaussianNB()
    NaiveBayes.fit(X_train,y_train)
    predicted_values = NaiveBayes.predict(X_test)
    x = metrics.accuracy_score(y_test, predicted_values)
    # print("Naive Bayes's Accuracy is: ", x)
    # print(classification_report(y_test,predicted_values))

    score = cross_val_score(NaiveBayes,X,y,cv=5)

    NB_pkl_filename = 'NBClassifier.pkl'
    NB_Model_pkl = open(NB_pkl_filename, 'wb')
    pickle.dump(NaiveBayes, NB_Model_pkl)
    NB_Model_pkl.close()
    
    data = np.array([[nitrogen, phosphorus, potash,
                      temp, humid, ph, rainfall]])
    prediction = NaiveBayes.predict(data)
    # print(prediction)
    
    return prediction


# print(predictedCropRecom(83, 45, 60, 28, 70.3, 7.0, 150.9))