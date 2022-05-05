from django.shortcuts import render
from .models import Covid, Article
import requests
import datetime
from .utils import get_scatter_plot, get_line_plot
from rest_framework import status, generics, viewsets
from .serializer import ArticleSerializer, CreateArticleSerializer, CovidModelSerializer, CreateCovidModelSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
import numpy as np

from covid_app import serializer

# Create your views here.

def get_covid_data(request):
    overall_covid_data = {}
    x = []
    y = []
    if 'name' in request.GET:
        date = request.GET['name']

        if Covid.objects.filter(report_date=date).count() == 0:
            url = "https://covid-19-statistics.p.rapidapi.com/reports/total"

            querystring = {"date":date}

            headers = {
	            "X-RapidAPI-Host": "covid-19-statistics.p.rapidapi.com",
	            "X-RapidAPI-Key": "5a522accafmsh4a078b3e82d1eecp195ebdjsn62e7ba20f597"
            }

            response = requests.request("GET", url, headers=headers, params=querystring)
            print(response)
            data = response.json()['data']

            covid_data = Covid(
                report_date = data['date'],
                confirmed = data['confirmed'],
                confirmed_diff = data['confirmed_diff'],
                deaths = data['deaths'],
                deaths_diff = data['deaths_diff'],
                recovered = data['recovered'],
                recovered_diff = data['recovered_diff'],
                active = data['active'],
                active_diff = data['active_diff'],
                fatality_rate = data['fatality_rate']
            )

            covid_data.save()
            if Covid.objects.filter(report_date__isnull = True).count() > 0:
                Covid.objects.filter(report_date__isnull = True).delete()
                
        overall_covid_data = Covid.objects.filter(report_date__isnull = False).order_by('report_date')

    if len(overall_covid_data) > 0:
        x = [x.confirmed for x in overall_covid_data]
        y = [y.deaths for y in overall_covid_data]

    # chart = get_plot(x, y)

    return render (request, 'coronavirus/covid.html', { "overall_covid_data": 
    overall_covid_data, "scatter_chart": get_scatter_plot(x, y), "line_chart": get_line_plot(x, y) })

def get_top_stories(request, *args, **kwargs):
    top_stories = {}
    if 'name' in request.GET:
        name = request.GET['name']

        url = "https://free-news.p.rapidapi.com/v1/search"

        querystring = {"q":name,"lang":"en"}

        headers = {
	        "X-RapidAPI-Host": "free-news.p.rapidapi.com",
	        "X-RapidAPI-Key": "5a522accafmsh4a078b3e82d1eecp195ebdjsn62e7ba20f597"
        }

        response = requests.request("GET", url, headers=headers, params=querystring)
        
        articles = response.json()['articles']

        for article in articles:

            if Article.objects.filter(link=article['link']).count() == 0:
                
                article_data = Article(
                    media = article['media'],
                    title = article['title'],
                    summary = article['summary'],
                    link = article['link'],
                    published_date = datetime.datetime.fromisoformat(article['published_date']),
                    query = response.json()['user_input']['q'],
                    rank = article['rank'],
                    topic = article['topic'],
                    country = article['country'],
                    score = article['_score']
                )

                article_data.save()

        top_stories = Article.objects.filter(query=name).order_by('-published_date')

    return render (request, 'coronavirus/stories.html', { "top_stories": 
    top_stories })

def article_detail(request, id):
    article = Article.objects.get(id = id)
    print(article)
    return render (
        request,
        'coronavirus/article_detail.html',
        {'article': article}
    )

def confirmed_predict(request):
    covid_data = Covid.objects.filter(report_date__isnull = False).order_by('report_date')
    X = np.array(range(len(covid_data))).reshape(-1, 1)
    y = np.array([y.confirmed for y in covid_data]).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_lin_pred = lin_reg.predict(X_train)
    lin_mse = mean_squared_error(y_train, y_lin_pred)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y_train, y_lin_pred)
    y_test_lin_pred = lin_reg.predict(X_test)
    test_lin_mse = mean_squared_error(y_test, y_test_lin_pred)
    test_lin_rmse = np.sqrt(test_lin_mse)
    test_lin_mae = mean_absolute_error(y_test, y_test_lin_pred)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    y_tree_pred = tree_reg.predict(X_train)
    tree_mse = mean_squared_error(y_train, y_tree_pred)
    tree_rmse = np.sqrt(tree_mse)
    y_test_tree_pred = tree_reg.predict(X_test)
    test_tree_mse = mean_squared_error(y_test, y_test_tree_pred)
    test_tree_rmse = np.sqrt(test_tree_mse)
    #test_tree_mae = mean_absolute_error(y_test, y_test_tree_pred)

    scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error")
    tree_rmse_scores = np.sqrt(-scores)

    test_scores = cross_val_score(tree_reg, X_test, y_test, scoring="neg_mean_squared_error")
    test_tree_rmse_scores = np.sqrt(-test_scores)

    lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error")
    lin_rmse_scores = np.sqrt(-lin_scores)

    test_lin_scores = cross_val_score(lin_reg, X_test, y_test, scoring="neg_mean_squared_error")
    test_lin_rmse_scores = np.sqrt(-test_lin_scores)

    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_train, y_train)

    y_forest_pred = forest_reg.predict(X_train)
    forest_mse = mean_squared_error(y_train, y_forest_pred)
    forest_rmse = np.sqrt(forest_mse)
    #forest_rmse

    y_test_forest_pred = forest_reg.predict(X_test)
    test_forest_mse = mean_squared_error(y_test, y_test_forest_pred)
    test_forest_rmse = np.sqrt(test_forest_mse)

    forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error")
    forest_rmse_scores = np.sqrt(-forest_scores)

    test_forest_scores = cross_val_score(forest_reg, X_test, y_test, scoring="neg_mean_squared_error")
    test_forest_rmse_scores = np.sqrt(-test_forest_scores)

    svm_reg = SVR(kernel="linear")
    svm_reg.fit(X_train, y_train)
    y_svm_pred = svm_reg.predict(X_train)
    svm_mse = mean_squared_error(y_train, y_svm_pred)
    svm_rmse = np.sqrt(svm_mse)

    y_test_svm_pred = svm_reg.predict(X_test)
    test_svm_mse = mean_squared_error(y_test, y_test_svm_pred)
    test_svm_rmse = np.sqrt(test_svm_mse)
    
    return render (request, 'coronavirus/covid_predict.html', { 'lin_rmse': lin_rmse, 'lin_mae': lin_mae, 'test_lin_rmse':
    test_lin_rmse, 'test_lin_mae': test_lin_mae, 'tree_rmse': tree_rmse, 'test_tree_rmse': test_tree_rmse, 'tree_rmse_scores': 
    tree_rmse_scores, 'test_tree_rmse_scores': test_tree_rmse_scores, 'tree_rmse_mean': tree_rmse_scores.mean(), 
    'test_tree_rmse_mean': test_tree_rmse_scores.mean(), 'tree_rmse_std': tree_rmse_scores.std(), 'test_tree_rmse_std': 
    test_tree_rmse_scores.std(), 'lin_rmse_scores': lin_rmse_scores, 'test_lin_rmse_scores': test_lin_rmse_scores, 
    'lin_rmse_mean': lin_rmse_scores.mean(), 'test_lin_rmse_mean': test_lin_rmse_scores.mean(), 'lin_rmse_std': 
    lin_rmse_scores.std(), 'test_lin_rmse_std': test_lin_rmse_scores.std(), 'forest_rmse': forest_rmse, 'test_forest_rmse': 
    test_forest_rmse, 'forest_rmse_scores': forest_rmse_scores, 'test_forest_rmse_scores': test_forest_rmse_scores, 
    'forest_rmse_mean': forest_rmse_scores.mean(), 'test_forest_rmse_mean': test_forest_rmse_scores.mean(), 'forest_rmse_std': 
    forest_rmse_scores.std(), 'test_forest_rmse_std': test_forest_rmse_scores.std(), 'svm_rmse': svm_rmse, 'test_svm_rmse': 
    test_svm_rmse })

def deaths_predict(request):
    covid_data = Covid.objects.filter(report_date__isnull = False).order_by('report_date')
    X = np.array(range(len(covid_data))).reshape(-1, 1)
    y = np.array([y.deaths for y in covid_data]).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_lin_pred = lin_reg.predict(X_train)
    lin_mse = mean_squared_error(y_train, y_lin_pred)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(y_train, y_lin_pred)
    y_test_lin_pred = lin_reg.predict(X_test)
    test_lin_mse = mean_squared_error(y_test, y_test_lin_pred)
    test_lin_rmse = np.sqrt(test_lin_mse)
    test_lin_mae = mean_absolute_error(y_test, y_test_lin_pred)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    y_tree_pred = tree_reg.predict(X_train)
    tree_mse = mean_squared_error(y_train, y_tree_pred)
    tree_rmse = np.sqrt(tree_mse)
    y_test_tree_pred = tree_reg.predict(X_test)
    test_tree_mse = mean_squared_error(y_test, y_test_tree_pred)
    test_tree_rmse = np.sqrt(test_tree_mse)
    #test_tree_mae = mean_absolute_error(y_test, y_test_tree_pred)

    scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error")
    tree_rmse_scores = np.sqrt(-scores)

    test_scores = cross_val_score(tree_reg, X_test, y_test, scoring="neg_mean_squared_error")
    test_tree_rmse_scores = np.sqrt(-test_scores)

    lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error")
    lin_rmse_scores = np.sqrt(-lin_scores)

    test_lin_scores = cross_val_score(lin_reg, X_test, y_test, scoring="neg_mean_squared_error")
    test_lin_rmse_scores = np.sqrt(-test_lin_scores)

    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X_train, y_train)

    y_forest_pred = forest_reg.predict(X_train)
    forest_mse = mean_squared_error(y_train, y_forest_pred)
    forest_rmse = np.sqrt(forest_mse)
    #forest_rmse

    y_test_forest_pred = forest_reg.predict(X_test)
    test_forest_mse = mean_squared_error(y_test, y_test_forest_pred)
    test_forest_rmse = np.sqrt(test_forest_mse)

    forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error")
    forest_rmse_scores = np.sqrt(-forest_scores)

    test_forest_scores = cross_val_score(forest_reg, X_test, y_test, scoring="neg_mean_squared_error")
    test_forest_rmse_scores = np.sqrt(-test_forest_scores)

    svm_reg = SVR(kernel="linear")
    svm_reg.fit(X_train, y_train)
    y_svm_pred = svm_reg.predict(X_train)
    svm_mse = mean_squared_error(y_train, y_svm_pred)
    svm_rmse = np.sqrt(svm_mse)

    y_test_svm_pred = svm_reg.predict(X_test)
    test_svm_mse = mean_squared_error(y_test, y_test_svm_pred)
    test_svm_rmse = np.sqrt(test_svm_mse)
    
    return render (request, 'coronavirus/deaths_predict.html', { 'lin_rmse': lin_rmse, 'lin_mae': lin_mae, 'test_lin_rmse':
    test_lin_rmse, 'test_lin_mae': test_lin_mae, 'tree_rmse': tree_rmse, 'test_tree_rmse': test_tree_rmse, 'tree_rmse_scores': 
    tree_rmse_scores, 'test_tree_rmse_scores': test_tree_rmse_scores, 'tree_rmse_mean': tree_rmse_scores.mean(), 
    'test_tree_rmse_mean': test_tree_rmse_scores.mean(), 'tree_rmse_std': tree_rmse_scores.std(), 'test_tree_rmse_std': 
    test_tree_rmse_scores.std(), 'lin_rmse_scores': lin_rmse_scores, 'test_lin_rmse_scores': test_lin_rmse_scores, 
    'lin_rmse_mean': lin_rmse_scores.mean(), 'test_lin_rmse_mean': test_lin_rmse_scores.mean(), 'lin_rmse_std': 
    lin_rmse_scores.std(), 'test_lin_rmse_std': test_lin_rmse_scores.std(), 'forest_rmse': forest_rmse, 'test_forest_rmse': 
    test_forest_rmse, 'forest_rmse_scores': forest_rmse_scores, 'test_forest_rmse_scores': test_forest_rmse_scores, 
    'forest_rmse_mean': forest_rmse_scores.mean(), 'test_forest_rmse_mean': test_forest_rmse_scores.mean(), 'forest_rmse_std': 
    forest_rmse_scores.std(), 'test_forest_rmse_std': test_forest_rmse_scores.std(), 'svm_rmse': svm_rmse, 'test_svm_rmse': 
    test_svm_rmse })

# def deaths_confirmed_predict(request):
#     covid_data = Covid.objects.all().order_by('report_date')
#     X = np.array([x.confirmed for x in covid_data]).reshape(-1, 1)
#     y = np.array([y.deaths for y in covid_data]).reshape(-1, 1)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
#     lin_reg = LinearRegression()
#     lin_reg.fit(X_train, y_train)
#     y_lin_pred = lin_reg.predict(X_train)
#     lin_mse = mean_squared_error(y_train, y_lin_pred)
#     lin_rmse = np.sqrt(lin_mse)
#     lin_mae = mean_absolute_error(y_train, y_lin_pred)

#     tree_reg = DecisionTreeRegressor(random_state=42)
#     tree_reg.fit(X_train, y_train)
#     y_tree_pred = tree_reg.predict(X_train)
#     tree_mse = mean_squared_error(y_train, y_tree_pred)
#     tree_rmse = np.sqrt(tree_mse)

#     scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error")
#     tree_rmse_scores = np.sqrt(-scores)

#     lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error")
#     lin_rmse_scores = np.sqrt(-lin_scores)

#     # forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
#     # forest_reg.fit(X_train, y_train.ravel())

#     # y_forest_pred = forest_reg.predict(X_train)
#     # forest_mse = mean_squared_error(y_train, y_forest_pred)
#     # forest_rmse = np.sqrt(forest_mse)
#     # forest_rmse

#     # forest_scores = cross_val_score(forest_reg, X_train, y_train.ravel(), scoring="neg_mean_squared_error")
#     # forest_rmse_scores = np.sqrt(-forest_scores)

#     svm_reg = SVR(kernel="linear")
#     svm_reg.fit(X_train, y_train)
#     y_svm_pred = svm_reg.predict(X_train)
#     svm_mse = mean_squared_error(y_train, y_svm_pred)
#     svm_rmse = np.sqrt(svm_mse)
    
#     return render (request, 'coronavirus/deaths_confirmed.html', { 'lin_rmse': lin_rmse, 'lin_mae': lin_mae, 
#     'tree_rmse': tree_rmse, 'tree_rmse_scores': tree_rmse_scores, 'tree_rmse_mean': tree_rmse_scores.mean(), 'tree_rmse_std': 
#     tree_rmse_scores.std(), 'lin_rmse_scores': lin_rmse_scores, 'lin_rmse_mean': lin_rmse_scores.mean(), 'lin_rmse_std': 
#     lin_rmse_scores.std(), 'svm_rmse': svm_rmse })

class CovidViewSet(generics.ListAPIView):
    queryset = Covid.objects.all()
    serializer_class = CovidModelSerializer

class ArticleViewSet(generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

class ArticleViewList(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

class CreateCovidModelView(APIView):
    serializer_class = CreateCovidModelSerializer

    def post(self, request, format=None):
        # if not self.request.session.exists(self.request.session.session_key):
        #     self.request.session.create()

        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            report_date = serializer.data.get('report_date')
            confirmed = serializer.data.get('confirmed')
            confirmed_diff = serializer.data.get('confirmed_diff')
            deaths = serializer.data.get('deaths')
            deaths_diff = serializer.data.get('deaths_diff')
            recovered = serializer.data.get('recovered')
            recovered_diff = serializer.data.get('recovered_diff')
            active = serializer.data.get('active')
            active_diff = serializer.data.get('active_diff')
            fatality_rate = serializer.data.get('fatality_rate')
            queryset = Covid.objects.filter(report_date=report_date)
            if queryset.exists():
                covid = queryset[0]
                covid.confirmed = confirmed
                covid.confirmed_diff = confirmed_diff
                covid.deaths = deaths
                covid.deaths_diff = deaths_diff
                covid.recovered = recovered
                covid.recovered_diff = recovered_diff
                covid.active = active
                covid.active_diff = active_diff
                covid.fatality_rate = fatality_rate
                covid.save(update_fields=['confirmed', 'confirmed_diff', 'deaths', 'deaths_diff', 'recovered', 
                'recovered_diff', 'active', 'active_diff', 'fatality_rate'])
                return Response(CovidModelSerializer(covid).data, status=status.HTTP_200_OK)
            else:
                covid = Covid(report_date=report_date, confirmed=confirmed, confirmed_diff=confirmed_diff, deaths=deaths, 
                deaths_diff=deaths_diff, recovered=recovered, recovered_diff=recovered_diff, active=active, active_diff=
                active_diff, fatality_rate=fatality_rate)
                covid.save()
                return Response(CovidModelSerializer(covid).data, status=status.HTTP_201_CREATED)

        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)

class CreateArticleView(APIView):
    serializer_class = CreateArticleSerializer

    def post(self, request, format=None):
        # if not self.request.session.exists(self.request.session.session_key):
        #     self.request.session.create()

        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            published_date = serializer.data.get('published_date')
            query = serializer.data.get('query')
            media = serializer.data.get('media')
            title = serializer.data.get('title')
            link = serializer.data.get('link')
            summary = serializer.data.get('summary')
            rank = serializer.data.get('rank')
            topic = serializer.data.get('topic')
            score = serializer.data.get('score')
            country = serializer.data.get('country')
            queryset = Article.objects.filter(link=link)
            if queryset.exists():
                article = queryset[0]
                article.published_date = published_date
                article.query = query
                article.media = media
                article.title = title
                article.score = score
                article.summary = summary
                article.rank = rank
                article.topic = topic
                article.country = country
                article.save(update_fields=['published_date', 'query', 'media', 'title', 'summary', 
                'rank', 'topic', 'score', 'country'])
                return Response(ArticleSerializer(article).data, status=status.HTTP_200_OK)
            else:
                article = Article(published_date=published_date, query=query, media=media, title=title, 
                summary=summary, rank=rank, topic=topic, score=score, country=country, link=link)
                article.save()
                return Response(ArticleSerializer(article).data, status=status.HTTP_201_CREATED)

        return Response({'Bad Request': 'Invalid data...'}, status=status.HTTP_400_BAD_REQUEST)