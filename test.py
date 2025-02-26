# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser


def test():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test'] 
    arg_dict['test_mode'] = True

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()

    # Build metrics
    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}

    # peak NRMSE metric data
    all_targets = []
    all_predictions = []
    top_k_nums = None

    count =0
    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            if arg_dict['cpu']:
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            prediction = model(input)

            # 收集数据
            all_targets.append(target)
            all_predictions.append(prediction.cpu())

            for metric, metric_func in metrics.items():
                if metric == 'peakNRMSE':
                    continue
                if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                    avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

            if arg_dict['plot_roc']:
                save_path = osp.join(arg_dict['save_path'], 'test_result')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_name = osp.splitext(osp.basename(label_path[0]))[0]
                save_path = osp.join(save_path, f'{file_name}.npy')
                output_final = prediction.float().detach().cpu().numpy()
                np.save(save_path, output_final)
                count +=1

            bar.update(1)

    # add peak NRMSE metric
    for metric, metric_func in metrics.items():
        if metric == 'peakNRMSE':
            print("\n===> peak NRMSE in {} samples, all targets {}, all predicitons {}, "
                  .format(len(dataset), len(all_targets), len(all_predictions)))

            import torch
            all_targets = torch.cat(all_targets, dim=0)  # [N, 256, 256, 2]
            all_predictions = torch.cat(all_predictions, dim=0)

            target_means = all_targets.mean(dim=(1, 2, 3))

            top_k_nums = int(len(target_means) * 0.05)  # 前 %5
            top_indices = torch.argsort(target_means, descending=True)[:top_k_nums]

            selected_targets = all_targets[top_indices]
            selected_predictions = all_predictions[top_indices]

            for selected_target, selected_prediction in zip(selected_targets, selected_predictions):
                target = selected_target.clone().unsqueeze(0)  # [1, 1, 256, 256]
                prediction = selected_prediction.clone().unsqueeze(0)

                if not metric_func(target.cpu(), prediction.cpu()) == 1:
                    avg_metrics[metric] += metric_func(target.cpu(), prediction.cpu())

    for metric, avg_metric in avg_metrics.items():
        if metric == 'peakNRMSE':
            print("===> Avg. {}: {:.4f}".format(metric, avg_metric / top_k_nums))
        else:
            print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))

    # eval roc&prc
    if arg_dict['plot_roc']:
        roc_metric, _ = build_roc_prc_metric(**arg_dict)
        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))


if __name__ == "__main__":
    test()
