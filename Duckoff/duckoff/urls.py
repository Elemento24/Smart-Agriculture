from django.contrib import admin
from django.urls import path
from duckoff import views
urlpatterns = [
    #  path("",views.index,name='home'),
    path("", views.index, name='index'),
    path("login", views.login, name='Login'),
    path("signup/", views.signup, name='signup'),
    path("croprecommendation", views.croprecommendation, name="croprecommendation"),
    path("plantdisease", views.plant_disease, name="plant disease")
]
