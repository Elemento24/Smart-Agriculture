# Generated by Django 2.2.16 on 2021-04-23 09:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('duckoff', '0003_user_age'),
    ]

    operations = [
        migrations.RenameField(
            model_name='user',
            old_name='District',
            new_name='City',
        ),
    ]
