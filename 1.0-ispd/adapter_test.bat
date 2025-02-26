@echo off

REM train
echo start testing

REM set python path to find routability_ir_drop_prediction module
set PYTHONPATH=E:\szx\Adapter-CircuitNet
REM superblue11_a
python -W ignore ..\test.py ^
    --task transfer_learning_adapter ^
    --pretrained .\models\teacher_student\superblue12\model_iters_20.pth ^
    --dataroot ..\data\target_datasets\superblue11_a\ ^
    --ann_file_test ..\data\target_datasets\superblue11_a.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\teacher_student\superblue11_test.log 2>&1
if %errorlevel% == 0 (
    echo superblue11_a done
) else (
    echo superblue11_a failed
)

REM superblue14
python -W ignore ..\test.py ^
    --task transfer_learning_adapter ^
    --pretrained .\models\teacher_student\superblue12\model_iters_20.pth ^
    --dataroot ..\data\target_datasets\superblue14\ ^
    --ann_file_test ..\data\target_datasets\superblue14.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\teacher_student\superblue14_test.log 2>&1
if %errorlevel% == 0 (
    echo superblue14 done
) else (
    echo superblue14 failed
)

REM superblue16_a
python -W ignore ..\test.py ^
    --task transfer_learning_adapter ^
    --pretrained .\models\teacher_student\superblue12\model_iters_20.pth ^
    --dataroot ..\data\target_datasets\superblue16_a\ ^
    --ann_file_test ..\data\target_datasets\superblue16_a.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\teacher_student\superblue16_a_test.log 2>&1
if %errorlevel% == 0 (
    echo superblue16_a done
) else (
    echo superblue16_a failed
)

REM superblue19
python -W ignore ..\test.py ^
    --task transfer_learning_adapter ^
    --pretrained .\models\teacher_student\superblue12\model_iters_20.pth ^
    --dataroot ..\data\target_datasets\superblue19\ ^
    --ann_file_test ..\data\target_datasets\superblue19.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > .\log\teacher_student\superblue19_test.log 2>&1
if %errorlevel% == 0 (
    echo superblue19 done
) else (
    echo superblue19 failed
)