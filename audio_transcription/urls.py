from django.urls import path
from .views import TranscriptionView
 
urlpatterns = [
    path('transcribe/', TranscriptionView.as_view(), name='transcribe'),
] 