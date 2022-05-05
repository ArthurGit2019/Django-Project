from django.db import models

# Create your models here.
class Covid(models.Model):
    report_date = models.CharField(max_length=20, blank=True, null=True)
    confirmed = models.IntegerField(blank=True, null=True)
    confirmed_diff = models.IntegerField(blank=True, null=True)
    deaths = models.IntegerField(blank=True, null=True)
    deaths_diff = models.IntegerField(blank=True, null=True)
    recovered = models.IntegerField(blank=True, null=True)
    recovered_diff = models.IntegerField(blank=True, null=True)
    active = models.IntegerField(blank=True, null=True)
    active_diff = models.IntegerField(blank=True, null=True)
    fatality_rate = models.DecimalField(max_digits=5, decimal_places=4, blank=True, null=True)
    
    def __str__(self):
        return self.report_date

class Article(models.Model):
    published_date = models.DateTimeField(auto_now_add=False)
    query = models.CharField(max_length=100, blank=True, null=True)
    media = models.CharField(max_length=200, blank=True, null=True)
    title = models.CharField(max_length=200, blank=True, null=True)
    link = models.CharField(max_length=200, blank=True, null=True)
    summary = models.CharField(max_length=4000, blank=True, null=True)
    rank = models.IntegerField(null=False, default=0)
    topic = models.CharField(max_length=20, blank=True, null=True)
    score = models.DecimalField(max_digits=8, decimal_places=6, null=False, default=10.000000)
    country = models.CharField(max_length=100, blank=True, null=True)

    def __datetime__(self):
        return self.published_date

