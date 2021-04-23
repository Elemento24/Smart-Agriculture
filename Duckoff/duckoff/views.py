import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from django.shortcuts import render
from django.http import HttpResponse
from django.http import Http404
from duckoff.models import user
from django.contrib import messages as message
from datetime import datetime
from django.contrib.messages import constants
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.messages import constants as messages
# Create your views here.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')
# import lazypredict
# from lazypredict.Supervised import LazyClassifier


def croprecommendation(request):

    if request.method == "POST":

        df = pd.read_csv('datasets/Crop_recommendation.csv')
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
        N = request.POST['nitrogen']
        phosphorous = request.POST['phosphorous']
        potash = request.POST['potash']
        temp = request.POST['temp']
        humid = request.POST['humid']
        ph = request.POST['ph']
        rain = request.POST['rainfall']

        def predictedCropRecom(nitrogen, phosphorus, potash,
                               temp, humid, ph, rainfall):

            NaiveBayes = GaussianNB()
            NaiveBayes.fit(X_train, y_train)
            predicted_values = NaiveBayes.predict(X_test)
            x = metrics.accuracy_score(y_test, predicted_values)
            # print("Naive Bayes's Accuracy is: ", x)
            # print(classification_report(y_test,predicted_values))

            score = cross_val_score(NaiveBayes, X, y, cv=5)

            NB_pkl_filename = 'models/NBClassifier.pkl'
            NB_Model_pkl = open(NB_pkl_filename, 'wb')
            pickle.dump(NaiveBayes, NB_Model_pkl)
            NB_Model_pkl.close()

            data = np.array([[nitrogen, phosphorus, potash,
                            temp, humid, ph, rainfall]])
            prediction = NaiveBayes.predict(data)
            print(prediction)
        predictedCropRecom(N, phosphorous, potash, temp, humid, ph, rain)

    else:
        return render(request, "croprecommendation.html")


def login(request):
    if request.method == 'POST':
        user_name = request.POST['Username']
        psw = request.POST['Password']
        # auth = User.objects.get(username=user_name)
        try:
            auth = User.objects.get(username=user_name)
        except User.DoesNotExist:
            auth = None
        if auth is not None:
            try:
                auth = authenticate(username=user_name, password=psw)
            except User.DoesNotExist:
                auth = None
            if auth is not None:
                return render(request, 'admin_login.html')
            else:
                return render(request, 'login.html')

        # auth = user.objects.get(Username=user_name)
        try:
            auth = user.objects.get(Username=user_name)
        except user.DoesNotExist:
            auth = None
        print(auth)
        if auth is not None and check_password(psw, auth.Password):
            return render(request, 'user_login.html')
        else:
            message.add_message(request, messages.INFO,
                                'Incorrect password/username !!!')
            return render(request, 'login.html')
    else:
        return render(request, 'login.html')


def signup(request):
    if request.method == 'POST':
        name = request.POST['name']
        age = request.POST['age']
        repsw = request.POST['re-password']
        psw = request.POST['password']
        if repsw != psw:
            message.add_message(request, messages.INFO, 'Paswword not matched')
            return render(request, 'signup.html')
        username = request.POST['username']
        state = request.POST['state']
        city = request.POST['city']
        User = user(Name=name, Username=username, Password=make_password(
            psw), Age=age, State=state, City=city)
        User.save()
        return render(request, 'login.html')
    else:
        return render(request, 'signup.html')
