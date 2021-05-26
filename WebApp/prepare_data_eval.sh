#! /bin/sh -e
echo Preparing data, please wait 30 minutes > ./log/evaluation_status.file
wait
unzip -o ../upload/$1.zip -d ../data/dcm_base/Eval$2/
wait
mv ../data/dcm_base/Eval$2/$1/* ../data/dcm_base/Eval$2/
wait
rm -rf ../data/dcm_base/Eval$2/$1/
wait
python3 ./DicomToNii-multi_eval.py evaluation_id.file
wait
python3 Task_eval.py evaluation_id.file eval_label.file evaluation_id.file taskid.file
wait
echo Data preparation done > ./log/evaluation_status.file