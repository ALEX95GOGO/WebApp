#!/bin/bash

cp $1/upload/$2_$3.zip /home/img/data/
unzip -o /home/img/data/$2_$3.zip 
python3 ../Task121_BreCW.py taskid.file