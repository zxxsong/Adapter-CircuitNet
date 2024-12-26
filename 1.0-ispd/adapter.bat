@echo off

REM train
echo start training

REM set python path to find routability_ir_drop_prediction module
set PYTHONPATH=E:\szx\Adapter-CircuitNet
REM superblue11_a
REM ignore warning
python -W ignore ..\train.py ^
    --task transfer_learning_adapter ^
    --save_path .\models\adapter_model\superblue12\ ^
    --pretrained .\models\pretrained\congestion_gpdl\model_iters_170000.pth ^
    --dataroot ..\data\target_datasets\superblue12\ ^
    --ann_file_train ..\data\target_datasets\superblue12.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\adapter\superblue12.log 2>&1
if %errorlevel% == 0 (
    echo superblue12 done
) else (
    echo superblue12 failed
)