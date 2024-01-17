import pandas as pd
import argparse
import math
import json


def elo_rank(scored_file, initial_score=1500, scaling_factor=400, max_gain=20):
    rank_scores = {
        'relevance': {},
        'fluency': {},
        'usefulness': {},
        'creativity': {}
    }

    df = pd.read_csv(scored_file)
    for row_id, row in df.iterrows():
        if not row['eval']:
            continue

        eval_result = json.loads(row['eval'])

        for metric in rank_scores:
            if row['model_A_name'] not in rank_scores[metric]:
                rank_scores[metric][row['model_A_name']] = initial_score
            if row['model_B_name'] not in rank_scores[metric]:
                rank_scores[metric][row['model_B_name']] = initial_score
            # current score
            r_a = rank_scores[metric][row['model_A_name']]
            r_b = rank_scores[metric][row['model_B_name']]

            # win expectation
            e_a = 1 / (1 + math.pow(10, (r_b-r_a)/scaling_factor))
            e_b = 1 / (1 + math.pow(10, (r_a-r_b)/scaling_factor))

            # update scores
            if eval_result['{}_win'.format(metric)] == 'A':
                s_a = 1
                s_b = 0
            elif eval_result['{}_win'.format(metric)] == 'B':
                s_a = 0
                s_b = 1
            else:
                s_a = s_b = 0.5

            r_a_new = r_a + max_gain * (s_a - e_a)
            r_b_new = r_b + max_gain * (s_b - e_b)

            rank_scores[metric][row['model_A_name']] = r_a_new
            rank_scores[metric][row['model_B_name']] = r_b_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare_result', type=str, default='output_test.csv')
    parser.add_argument('--initial_score', type=int, default=1500)
    parser.add_argument('--scaling_factor', type=int, default=400)
    parser.add_argument('--max_gain', type=int, default=20)
    args = parser.parse_args()

    elo_rank(
        args.compare_result,
        initial_score=args.initial_score,
        scaling_factor=args.scale_factor,
        max_gain=args.max_gain
    )