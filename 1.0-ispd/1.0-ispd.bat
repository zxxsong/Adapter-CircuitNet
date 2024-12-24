@echo off

REM train
echo start training

REM set python path to find routability_ir_drop_prediction module
set PYTHONPATH=E:\szx\Adapter-CircuitNet
REM superblue11_a
REM ignore warning
python -W ignore ..\train.py ^
    --task transfer_learning_adapter ^
    --save_path .\models\adapter_model\superblue11_a\ ^
    --pretrained .\models\pretrained\congestion_gpdl\model_iters_170000.pth ^
    --dataroot ..\data\target_datasets\superblue11_a\ ^
    --ann_file_train ..\data\target_datasets\superblue11_a.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\superblue11_a.log 2>&1
if %errorlevel% == 0 (
    echo superblue11_a done
) else (
    echo superblue11_a failed
)

REM superblue14
python -W ignore ..\train.py ^
    --task transfer_learning_adapter ^
    --save_path .\models\adapter_model\superblue14\ ^
    --pretrained .\models\pretrained\congestion_gpdl\model_iters_170000.pth ^
    --dataroot ..\data\target_datasets\superblue14\ ^
    --ann_file_train ..\data\target_datasets\superblue14.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\superblue14.log 2>&1
if %errorlevel% == 0 (
    echo superblue14 done
) else (
    echo superblue14 failed
)

REM superblue16_a
python -W ignore ..\train.py ^
    --task transfer_learning_adapter ^
    --save_path .\models\adapter_model\superblue16_a\ ^
    --pretrained .\models\pretrained\congestion_gpdl\model_iters_170000.pth ^
    --dataroot ..\data\target_datasets\superblue16_a\ ^
    --ann_file_train ..\data\target_datasets\superblue16_a.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\superblue16_a.log 2>&1
if %errorlevel% == 0 (
    echo superblue16_a done
) else (
    echo superblue16_a failed
)

REM superblue19
python -W ignore ..\train.py ^
    --task transfer_learning_adapter ^
    --save_path .\models\adapter_model\superblue19\ ^
    --pretrained .\models\pretrained\congestion_gpdl\model_iters_170000.pth ^
    --dataroot ..\data\target_datasets\superblue19\ ^
    --ann_file_train ..\data\target_datasets\superblue19.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu  > .\log\superblue19.log 2>&1
if %errorlevel% == 0 (
    echo superblue19 done
) else (
    echo superblue19 failed
)

REM test
echo start testing