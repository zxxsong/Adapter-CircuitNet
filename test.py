# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np

from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric, topk_normalized_root_mse, topk_mse
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
    peak_details = {p: 0 for p in [0.005, 0.01, 0.02, 0.05]}
    mse_details = {m: 0 for m in [0.02, 0.05, 0.1]}

    count =0
    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            if arg_dict['cpu']:
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            prediction = model(input)

            for metric, metric_func in metrics.items():
                if metric == 'peakNRMSE': # no need to input_converter
                    # [1, 1, 256, 256] -> [256, 256]
                    target_tensor = target.cpu().detach()
                    pred_tensor = prediction.squeeze(1).cpu().detach()
                    if len(target_tensor.shape) > 2:
                        target_tensor = target_tensor.squeeze()
                    if len(pred_tensor.shape) > 2:
                        pred_tensor = pred_tensor.squeeze()
                    assert target_tensor.shape == pred_tensor.shape, f"Shape mismatch: {target_tensor.shape} vs {pred_tensor.shape}"

                    target_np = target_tensor.numpy()
                    pred_np = pred_tensor.numpy()
                    k_percentages = [0.005, 0.01, 0.02, 0.05]
                    peak_nrmse_values = {}
                    for topk_percent in k_percentages:
                        peak_nrmse_value = topk_normalized_root_mse(target_np, pred_np, topk_percent=topk_percent)
                        peak_nrmse_values[topk_percent] = peak_nrmse_value

                    peak_nrmse_mean_value = np.mean(list(peak_nrmse_values.values()))

                    if not peak_nrmse_mean_value == 1:
                        avg_metrics[metric] += peak_nrmse_mean_value
                        for p in peak_details:
                            peak_details[p] += peak_nrmse_values[p]

                elif metric == "MSE": # no need to input_converter
                    # [1, 1, 256, 256] -> [256, 256]
                    target_tensor = target.cpu().detach()
                    pred_tensor = prediction.squeeze(1).cpu().detach()
                    if len(target_tensor.shape) > 2:
                        target_tensor = target_tensor.squeeze()
                    if len(pred_tensor.shape) > 2:
                        pred_tensor = pred_tensor.squeeze()
                    assert target_tensor.shape == pred_tensor.shape, f"Shape mismatch: {target_tensor.shape} vs {pred_tensor.shape}"

                    target_np = target_tensor.numpy()
                    pred_np = pred_tensor.numpy()

                    mse_percentages = [0.02, 0.05, 0.1]
                    mse_values = {}
                    for mse_percent in mse_percentages:
                        mse_value = topk_mse(target_np, pred_np, mse_percent=mse_percent)
                        mse_values[mse_percent] = mse_value

                    for m in mse_details:
                        if not mse_values[m] == 1:
                            mse_details[m] += mse_values[m]

                elif not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
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

    for metric, avg_metric in avg_metrics.items():
        if metric == 'peakNRMSE':
            print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))
            for p in peak_details:
                print("===> Avg. peakNRMSE at top {}%: {:.4f}".format(p*100, peak_details[p] / len(dataset)))
        elif metric == 'MSE':
            for m in mse_details:
                print("===> Avg. MSE at top {}%: {:.4f}".format(m*100, mse_details[m] / len(dataset)))
        else:
            print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))

    # eval roc&prc
    if arg_dict['plot_roc']:
        roc_metric, _ = build_roc_prc_metric(**arg_dict)
        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))


if __name__ == "__main__":
    test()
