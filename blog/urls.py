from django.urls import path
from .views import TitleSuggestionView
 
urlpatterns = [
    path('suggest-title/', TitleSuggestionView.as_view(), name='suggest-title'),
] 