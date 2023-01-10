#! /bin/sh -e

export nnUNet_raw_data_base="/home/zhuoli/data/nnUNet_raw_data_base/"
export nnUNet_preprocessed="/home/zhuoli/data/nnUNet_preprocessed/"
export RESULTS_FOLDER="/home/zhuoli/data/nnUNet_trained_models/"
wait
echo Planning > ./log/training_status.file
wait
python3 ./Taskxxx_multi.py taskid.file train_label.file
wait
nnUNet_plan_and_preprocess -t $1 > ./log/plan.file 2>&1 &
wait
echo Training > ./log/training_status.file
wait
echo nnUNet_train $3 nnUNetTrainerV2_noDeepSupervision $1 0 --npz > ./log/training_log.file
nnUNet_train $3 nnUNetTrainerV2_noDeepSupervision $1 0 --npz > ./log/train.file 2>&1 &
wait
echo Predicting > ./log/training_status.file
wait
nnUNet_predict -i $4/nnUNet_raw_data/Task$1_$2/imagesTs/ -o $4/result/Task$1_$2 -t $1 -m $3 -f 0 -tr nnUNetTrainerV2_noDeepSupervision --disable_tta -chk model_best > ./log/predict.file 2>&1 &
wait
echo Training Done > ./log/training_status.file
