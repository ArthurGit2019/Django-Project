# Generated by Django 2.2.12 on 2022-04-29 05:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covid_app', '0004_auto_20220429_0532'),
    ]

    operations = [
        migrations.AlterField(
            model_name='article',
            name='summary',
            field=models.CharField(max_length=40, null=True),
        ),
    ]
