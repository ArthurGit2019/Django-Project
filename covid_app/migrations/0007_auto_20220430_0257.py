# Generated by Django 2.2.12 on 2022-04-30 02:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covid_app', '0006_auto_20220429_0544'),
    ]

    operations = [
        migrations.AddField(
            model_name='article',
            name='query',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
        migrations.AlterField(
            model_name='article',
            name='summary',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
    ]