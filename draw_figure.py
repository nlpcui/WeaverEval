import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


# data = np.random.random([5, 5])
# print(data)
# fig = sns.heatmap(data=data, square=True, center=0.5, cmap='RdBu_r', annot=True, xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['a', 'b', 'c', 'd', 'e'])
# plt.title('test')
# plt.show()


def draw_heatmap(input_file, metric, model_names, center, cmap, pad=0):
    assert metric in ['fluency', 'relevance', 'creativity', 'usefulness', 'overall']

    win_rate = {}
    df = pd.read_csv(input_file)
    for row_id, row in df.iterrows():
        if row['model_A_name'] not in win_rate:
            win_rate[row['model_A_name']] = {}
        if row['model_B_name'] not in win_rate[row['model_A_name']]:
            win_rate[row['model_A_name']][row['model_B_name']] = [0, 0, 0]  # win, same, loss

        if row['model_B_name'] not in win_rate:
            win_rate[row['model_B_name']] = {}
        if row['model_A_name'] not in win_rate[row['model_B_name']]:
            win_rate[row['model_B_name']][row['model_A_name']] = [0, 0, 0]

        compare_result = json.loads(row['eval'])
        if compare_result[metric] == '左边好':
            win_rate[row['model_A_name']][row['model_B_name']][0] += 1
            win_rate[row['model_B_name']][row['model_A_name']][2] += 1
        elif compare_result[metric] == '右边好':
            win_rate[row['model_A_name']][row['model_B_name']][2] += 1
            win_rate[row['model_B_name']][row['model_A_name']][0] += 1
        else:
            win_rate[row['model_A_name']][row['model_B_name']][1] += 1
            win_rate[row['model_B_name']][row['model_A_name']][1] += 1

    if not model_names:
        model_names = list(win_rate.keys())

    matrix = np.zeros([len(model_names), len(model_names)])

    for row_id, row_model in enumerate(model_names):
        for column_id, column_model in enumerate(model_names):
            if row_model == column_model:
                matrix[row_id][column_id] = pad
                continue
            matrix[row_id][column_id] = round(win_rate[row_model][column_model][0] / (win_rate[row_model][column_model][0] + win_rate[row_model][column_model][2]), 2)  # drop tie

    sns.set_context({"figure.figsize": (8, 8)})
    sns.heatmap(data=matrix, square=True, center=center, cmap=cmap, annot=True, xticklabels=model_names, yticklabels=model_names)
    plt.title(metric)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='output_test.csv')
    parser.add_argument('--model_names', type=str, default='')
    parser.add_argument('--metric', type=str)
    parser.add_argument('--pad', type=float, default=.0)
    parser.add_argument('--center', type=float, default=0.5)
    parser.add_argument('--cmap', type=str, default='RdBu_r')

    args = parser.parse_args()
    draw_heatmap(
        input_file=args.input_file,
        model_names=args.model_names.split(','),
        metric=args.metric,
        center=args.center,
        cmap=args.cmap,
        pad=args.pad
    )