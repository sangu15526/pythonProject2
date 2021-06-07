from django.db import models
from django.contrib.auth.models import User


# Create your models here.


class last_login(models.Model):
    username = models.CharField(max_length=20)
    password = models.CharField(max_length=20)

    def __str__(self):
        return self.username


class attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    Time = models.TimeField()

    def __str__(self):
        return str(self.user) + " " + str(self.date) + " " + str(self.Time)


class attention(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    emotion = models.CharField(max_length=15)
    date = models.DateField()
    Time = models.TimeField()

    def __str__(self):
        return str(self.user) + " " + str(self.emotion) + " " + str(self.date) + " " + str(self.Time)


class class_hour(models.Model):
    faculty = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()


class reports(models.Model):
    class_hour = models.ForeignKey(class_hour, on_delete=models.CASCADE)
    overview = models.IntegerField()
    review = models.CharField(max_length=20)
    engaged_count = models.IntegerField()
    confusion_count = models.IntegerField()
    bored_count = models.IntegerField()
    frustrated_count = models.IntegerField()
