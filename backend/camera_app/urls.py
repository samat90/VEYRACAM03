from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('history/', views.history_page, name='history_page'),
    path('process-frame/', views.process_frame, name='process_frame'),
    path('stop-session/', views.stop_session, name='stop_session'),
    path('api/history/', views.session_history, name='session_history'),
    path('api/sessions/', views.session_list, name='session_list'),
]
