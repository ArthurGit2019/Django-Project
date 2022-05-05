from django.contrib import admin
from .models import Article, Covid

# Register your models here.
admin.site.register(Covid)
admin.site.register(Article)