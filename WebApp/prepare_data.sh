#! /bin/sh -e
echo Preparing data, please wait 30 minutes > ./log/prepare_data_status.file
wait
unzip -o ../upload/$1.zip -d ../data/dcm_base/Task$2/
wait
mv ../data/dcm_base/Task$2/$1/* ../data/dcm_base/Task$2/
wait
rm -rf ../data/dcm_base/Task$2/$1/
wait
python3 ./DicomToNii-multi.py taskid.file
wait
echo Data prepare done > ./log/prepare_data_status.file