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
import requests
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

        ph = request.POST['ph']
        rain = request.POST['rainfall']
        city = request.POST['city']

        def weather_fetch(city_name):
            """
            Fetch and returns the temperature and humidity of a city
            :params: city_name
            :return: temperature, humidity
            """
            api_key = "9d7cde1f6d07ec55650544be1631307e"
            base_url = "http://api.openweathermap.org/data/2.5/weather?"

            complete_url = base_url + "appid=" + api_key + "&q=" + city_name
            response = requests.get(complete_url)
            x = response.json()

            if x["cod"] != "404":
                y = x["main"]

                temperature = round((y["temp"] - 273.15), 2)
                humidity = y["humidity"]
                return temperature, humidity
            else:
                return None

        temp, humid = weather_fetch(city)
        # print(temperatue, humidity)
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
        user_name = request.POST['username']
        state = request.POST['state']
        city = request.POST['city']
        repsw = request.POST['re-password']
        psw = request.POST['password']
        try:
            auth = user.objects.get(Username=user_name)
        except user.DoesNotExist:
            auth = None

        if auth is not None:
            message.add_message(request, messages.INFO, 'username exsits')
            return render(request, 'signup.html')
        try:
            auth = User.objects.get(username=user_name)
        except user.DoesNotExist:
            auth = None
        if auth is not None:
            message.add_message(request, messages.INFO, 'username exsits')
            return render(request, 'signup.html')
        if repsw != psw:
            message.add_message(request, messages.INFO, 'Paswword not matched')
            return render(request, 'signup.html')

        user_ = user(Name=name, Username=user_name, Password=make_password(
            psw), Age=age, State=state, City=city)
        user_.save()
        return render(request, 'login.html')
    else:
        return render(request, 'signup.html')
