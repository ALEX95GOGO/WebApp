from __future__ import absolute_import
from celery import shared_task
import subprocess
import os
import re
from WebApp import settings
            
@shared_task(bind=True)
def run_script(self):
    with open('./log/taskid.file', "r") as f:
        data = f.read()
        s1 = re.split(' ', data)
        print(s1[0])
        print(s1[1])
        print('base dir')
        print(settings.BASE_DIR)
    subprocess.run('bash prepare_data.sh {} {} {}'.format(settings.BASE_DIR, s1[0], s1[1]), shell=True)
    return print('done')