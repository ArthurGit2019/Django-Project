# Generated by Django 2.2.12 on 2022-04-29 05:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covid_app', '0003_article'),
    ]

    operations = [
        migrations.AlterField(
            model_name='article',
            name='summary',
            field=models.CharField(blank=True, max_length=40, null=True),
        ),
    ]
