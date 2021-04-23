# Generated by Django 2.2.16 on 2021-04-23 09:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('duckoff', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='user',
            old_name='password',
            new_name='Password',
        ),
        migrations.RemoveField(
            model_name='user',
            name='age',
        ),
        migrations.AddField(
            model_name='user',
            name='District',
            field=models.CharField(default='', max_length=120),
        ),
        migrations.AddField(
            model_name='user',
            name='State',
            field=models.CharField(default='', max_length=120),
        ),
    ]
