# Generated by Django 2.2.16 on 2021-04-23 09:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('duckoff', '0002_auto_20210423_0940'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='Age',
            field=models.IntegerField(default=0),
        ),
    ]
