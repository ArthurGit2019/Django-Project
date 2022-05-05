from rest_framework import serializers
from .models import Covid, Article


class CovidModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Covid
        fields = '__all__'

class CreateCovidModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Covid
        fields = '__all__'

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = '__all__'

class CreateArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = '__all__'