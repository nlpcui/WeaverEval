import json
import math

import pandas as pd
from openai import OpenAI
import random
import logging
import argparse
from tqdm import tqdm

openai_key = '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'


def find_between(string, prefix, suffix):

    start = string.find(prefix) if prefix else 0
    end = string.find(suffix) if suffix else len(string)
    if start == -1 or end == -1:
        return None
    return string[start+len(prefix):end].strip()


def gpt4_evaluate(input_file, output_file, prompt_file, model_name='gpt-4-1106-preview', temperature=0.3, max_try=3):

    data_frame = pd.read_csv(input_file)
    pairs = {}
    for idx, row in data_frame.iterrows():
        pairs[row['id']] = {
            'domain': row['domain'],
            'task': row['task'],
            'instruction': row['instruction'],
            'model_A_name': row['model_A_name'],
            'model_A_output': row['model_A_output'],
            'model_B_name': row['model_B_name'],
            'model_B_output': row['model_B_output'],
            'eval': None
        }

    with open(prompt_file, 'r') as fp:
        prompt_template = ''.join(fp.readlines())
    client = OpenAI(api_key=openai_key)

    for pair_id in tqdm(pairs):
        pair = pairs[pair_id]
        content = prompt_template.format(
            domain=pair['domain'],
            task=pair['task'],
            instruction=pair['instruction'],
            model_A_output=pair['model_A_output'],
            model_B_output=pair['model_B_output']
        )
        messages = [{'role': 'user', 'content': content}]

        for req_id in range(max_try):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature
                )
                response_content = response.choices[0].message.content
                rationale = find_between(response_content, '[Analyses]', '[Comparison]')
                relevance_win = find_between(response_content, '* Relevance:', '* Fluency:')
                fluency_win = find_between(response_content, '* Fluency:', '* Usefulness:')
                usefulness_win = find_between(response_content, '* Usefulness:', '* Creativity:')
                creativity_win = find_between(response_content, '* Creativity:', '* Overall:')
                overall_win = find_between(response_content, '* Overall:', None)
                results = [rationale, relevance_win, fluency_win, usefulness_win, creativity_win, overall_win]
                print(pair_id, results[1:])
                print('analysis', rationale)
                exit(1)
                assert all(results)
                for res in results:
                    assert res in ['A', 'B', 'Same']
                pair['eval'] = json.dumps({
                    'rationale': rationale,
                    'relevance_win': relevance_win,
                    'fluency_win': fluency_win,
                    'usefulness_win': usefulness_win,
                    'creativity_win': creativity_win,
                    'overall_win': overall_win
                })
                break
            except Exception:
                continue

    scored_df = pd.DataFrame(pairs)
    scored_df.to_csv(output_file)


def elo_rank(scored_file, initial_score=1500, scaling_factor=400, k=20):
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

            r_a_new = r_a + k * (s_a - e_a)
            r_b_new = r_b + k * (s_b - e_b)

            rank_scores[metric][row['model_A_name']] = r_a_new
            rank_scores[metric][row['model_B_name']] = r_b_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='output_test.csv')
    parser.add_argument('--output_file', type=str, default='output_test_scored.csv')
    parser.add_argument('--prompt_file', type=str, default='prompts/eval_prompt.txt')
    args = parser.parse_args()

    gpt4_evaluate(
        input_file=args.input_file,
        output_file=args.output_file,
        prompt_file=args.prompt_file
    )
    elo_rank(args.output_file)