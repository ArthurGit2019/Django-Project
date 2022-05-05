from django.urls import path, include
from covid_app import views
# from covid_app.serializer import CovidModelSerializer
# from .models import Covid
# import numpy as np
from rest_framework import routers


# router = routers.DefaultRouter()
# router.register(r'covid', CovidModelSerializer, basename='MyModel')
router = routers.DefaultRouter()
router.register(r'stories', views.ArticleViewList)
#covid_data = Covid.objects.all().order_by('report_date')

urlpatterns = [
    path('', views.get_covid_data, name = "get_covid_data"),
    path('covid.html', views.get_covid_data, name = "get_covid_data"),
    path('stories.html', views.get_top_stories),
    path('articles/<int:id>/',views.article_detail),
    path('covid_predict.html', views.confirmed_predict),
    # path('covid_predict.html#confirmed_cases', views.confirmed_predict),
    path('deaths_predict.html', views.deaths_predict),
    #path('deaths_confirmed.html', views.deaths_confirmed_predict),
    path('covid', views.CovidViewSet.as_view()),
    path('create-covid-data', views.CreateCovidModelView.as_view()),
    path('article', views.ArticleViewSet.as_view()),
    path('create-article', views.CreateArticleView.as_view()),
    path('', include(router.urls)),
]