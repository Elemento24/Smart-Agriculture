from django.db import models

# Create your models here.


class plant_disease_image(models.Model):
    image = models.ImageField()
    date = models.DateTimeField(auto_now=True)


class user(models.Model):
    Name = models.CharField(max_length=120)
    Username = models.CharField(max_length=20)
    Password = models.CharField(max_length=20)
    Age = models.IntegerField(default=0)
    State = models.CharField(max_length=120, default="")
    City = models.CharField(max_length=120, default="")

    def __str__(self):
        return self.Username
