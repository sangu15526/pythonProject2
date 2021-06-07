from django.urls import path
from .views import *

urlpatterns = [
    path('', register, name="register"),
    path('home/', home, name="home"),
    path('login-h/', loginpage, name="login-h"),
    path('register/', register, name="register"),
    path('logout-h/', logout_User, name="logout-h"),
    path('add_face/',add_face,name="add_face"),
    path('create_model/',create_model,name='create_model'),
    path('attend/',attend,name='attend'),
    path('total/',total_attendance,name='total'),
    path('staff-home/',staff_home, name='staff-home'),
    path('view_review/',view_review,name='view_review'),
]
