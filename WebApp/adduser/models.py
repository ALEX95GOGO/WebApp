from django.db import models

class User(models.Model):
    def __init__(self, _id=None, _username=None, _password=None):
        self.id = _id
        self.username = _username
        self.password = _password
    
