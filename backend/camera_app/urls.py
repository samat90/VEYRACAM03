from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('history/', views.history_page, name='history_page'),
    path('process-frame/', views.process_frame, name='process_frame'),
    path('pause-session/', views.pause_session, name='pause_session'),
    path('stop-session/', views.stop_session, name='stop_session'),
    path('api/history/', views.session_history, name='session_history'),
    path('api/sessions/', views.session_list, name='session_list'),
    path('api/export/<int:session_id>/', views.export_session, name='export_session'),
]
