import os
import csv

# 目录列表
dirs = ['superblue11_a', 'superblue12', 'superblue14', 'superblue16_a', 'superblue19']

# 遍历每个目录
for dir_name in dirs:
    feature_dir = os.path.join(dir_name, 'feature')
    label_dir = os.path.join(dir_name, 'label')

    # 获取 feature 和 label 目录中的文件名
    feature_files = sorted(os.listdir(feature_dir))
    label_files = sorted(os.listdir(label_dir))

    # 确保 feature 和 label 文件数量一致
    if len(feature_files) != len(label_files):
        print(f"Warning: feature and label files count mismatch in {dir_name}")
        continue

    # 打开一个新的 CSV 文件进行写入
    with open(f'{dir_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        # writer.writerow(['Feature', 'Label'])

        # 遍历文件名并写入到 CSV 文件
        for feature_file, label_file in zip(feature_files, label_files):
            feature_path = os.path.join('feature', feature_file).replace('\\', '/')
            label_path = os.path.join('label', label_file).replace('\\', '/')
            # 写入对应的 feature 和 label 路径
            writer.writerow([feature_path, label_path])

    print(f"{dir_name}_file.csv generated successfully.")
