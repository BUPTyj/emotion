"""emotionweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from emotion import views

urlpatterns = [
    path('', views.index, name='index'),
    path('inf', views.inference, name='inference'),
    path('train', views.training, name='training'),
    path('start_training/', views.start_training, name='start_training'),
    path('stop_training/', views.stop_training, name='stop_training'),
    path('get_training_status/', views.get_training_status, name='get_training_status'),
    path('training_stream/', views.training_stream, name='training_stream'),
    path('get_training_results/', views.get_training_results, name='get_training_results'),
    path('test', views.testing),
]
