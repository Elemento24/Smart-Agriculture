# Generated by Django 2.2.16 on 2021-04-24 03:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('duckoff', '0004_auto_20210423_0946'),
    ]

    operations = [
        migrations.CreateModel(
            name='plant_disease_image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='')),
                ('date', models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
