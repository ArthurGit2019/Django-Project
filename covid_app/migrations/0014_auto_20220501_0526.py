# Generated by Django 2.2.12 on 2022-05-01 05:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covid_app', '0013_auto_20220430_0355'),
    ]

    operations = [
        migrations.AlterField(
            model_name='covid',
            name='report_date',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
    ]
