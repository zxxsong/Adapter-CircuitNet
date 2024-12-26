@echo off

REM train
echo start training

REM set python path to find routability_ir_drop_prediction module
set PYTHONPATH=D:\EDA\Adapter-CircuitNet
REM superblue12
python ..\train.py ^
    --task transfer_learning_freeze ^
    --save_path .\models\freeze_model\superblue12\ ^
    --pretrained .\models\pretrained\congestion_gpdl\model_iters_170000.pth ^
    --dataroot ..\data\target_datasets\superblue12\ ^
    --ann_file_train ..\data\target_datasets\superblue12.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\freeze\superblue12.log 2>&1
if %errorlevel% == 0 (
    echo superblue12 done
) else (
    echo superblue12 failed
)
