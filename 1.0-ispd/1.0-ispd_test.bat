@echo off

REM train
echo start testing

REM set python path to find routability_ir_drop_prediction module
set PYTHONPATH=E:\szx\CircuitNet
REM superblue11_a
python test.py ^
    --task congestion_transfer_learning ^
    --pretrained .\models\student_model\superblue11_a\model_iters_200.pth ^
    --dataroot .\data\target_dataset\transfer_learning_datasets\superblue12\ ^
    --ann_file_test .\data\target_dataset\transfer_learning_datasets\superblue12.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > superblue11_test.txt 2>&1
if %errorlevel% == 0 (
    echo superblue11_a done
) else (
    echo superblue11_a failed
)

REM superblue14
python test.py ^
    --task congestion_transfer_learning ^
    --pretrained .\models\student_model\superblue14\model_iters_200.pth ^
    --dataroot .\data\target_dataset\transfer_learning_datasets\superblue12\ ^
    --ann_file_test .\data\target_dataset\transfer_learning_datasets\superblue12.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > superblue14_test.txt 2>&1
if %errorlevel% == 0 (
    echo superblue14 done
) else (
    echo superblue14 failed
)

REM superblue16_a
python test.py ^
    --task congestion_transfer_learning ^
    --pretrained .\models\student_model\superblue16_a\model_iters_200.pth ^
    --dataroot .\data\target_dataset\transfer_learning_datasets\superblue12\ ^
    --ann_file_test .\data\target_dataset\transfer_learning_datasets\superblue12.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > superblue16_a_test.txt 2>&1
if %errorlevel% == 0 (
    echo superblue16_a done
) else (
    echo superblue16_a failed
)

REM superblue19
python test.py ^
    --task congestion_transfer_learning ^
    --pretrained .\models\student_model\superblue19\model_iters_200.pth ^
    --dataroot .\data\target_dataset\transfer_learning_datasets\superblue12\ ^
    --ann_file_test .\data\target_dataset\transfer_learning_datasets\superblue12.csv ^
    --dataset_type SuperBlueDataset ^
    --cpu > superblue19_test.txt 2>&1
if %errorlevel% == 0 (
    echo superblue19 done
) else (
    echo superblue19 failed
)