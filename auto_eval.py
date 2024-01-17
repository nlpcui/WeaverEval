import json
import math

import pandas as pd
from openai import OpenAI
import random
import logging
import argparse
from tqdm import tqdm

openai_key = ''


def find_between(string, prefix, suffix):

    start = string.find(prefix) if prefix else 0
    end = string.find(suffix) if suffix else len(string)
    if start == -1 or end == -1:
        return None
    return string[start+len(prefix):end].strip()


class GPT4Eval:
    def __init__(self, model_name, eval_prompt_pair, eval_prompt_single, temperature, max_try):
        self.model_name = model_name
        self.temperature = temperature
        self.max_try = max_try

        self.client = OpenAI(api_key=openai_key)

        with open(eval_prompt_pair, 'r') as fp:
            self.eval_prompt_pair_template = ''.join(fp.readlines())
        with open(eval_prompt_single, 'r') as fp:
            self.eval_prompt_single_template = ''.join(fp.readlines())

    def call_model(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )
        response_content = response.choices[0].message.content
        return response_content

    def eval_pair(self, input_file, output_file):
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

        client = OpenAI(api_key=openai_key)

        for pair_id in tqdm(pairs):
            pair = pairs[pair_id]
            content = self.eval_prompt_pair_template.format(
                domain=pair['domain'],
                task=pair['task'],
                instruction=pair['instruction'],
                model_A_output=pair['model_A_output'],
                model_B_output=pair['model_B_output']
            )

            for req_id in range(self.max_try):
                try:
                    response_content = self.call_model(content)
                    rationale = find_between(response_content, '[Analyses]', '[Comparison]')
                    relevance_win = find_between(response_content, '* Relevance:', '* Fluency:')
                    fluency_win = find_between(response_content, '* Fluency:', '* Usefulness:')
                    usefulness_win = find_between(response_content, '* Usefulness:', '* Creativity:')
                    creativity_win = find_between(response_content, '* Creativity:', '* Overall:')
                    overall_win = find_between(response_content, '* Overall:', None)
                    results = [rationale, relevance_win, fluency_win, usefulness_win, creativity_win, overall_win]
                    # print(pair_id, results[1:])
                    # print('analysis', rationale)
                    # exit(1)
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

    def eval_single(self, input_file, output_file):
        data = []
        df = pd.read_csv(input_file)
        for row_id, row in tqdm(df.iterrows()):
            for req_id in range(self.max_try):
                content = self.eval_prompt_single_template.format(
                    domain=row['domain'],
                    task=row['task'],
                    instruction=row['instruction'],
                    model_output=row['model_output']
                )
                try:
                    response_content = self.call_model(prompt=content)

                    rationale = find_between(response_content, '[Analyses]', '[Scores]')
                    relevance_score = float(find_between(response_content, '* Relevance:', '* Fluency:'))
                    fluency_score = float(find_between(response_content, '* Fluency:', '* Usefulness:'))
                    usefulness_score = float(find_between(response_content, '* Usefulness:', '* Creativity:'))
                    creativity_score = float(find_between(response_content, '* Creativity:', '* Overall:'))
                    overall_score = float(find_between(response_content, '* Overall:', None))

                    data.append({
                        'id': row['id'],
                        'domain': row['domain'],
                        'task': row['task'],
                        'model_name': row['model_name'],
                        'model_output': row['model_output'],
                        'scores': json.dumps({
                            'relevance': relevance_score,
                            'fluency': fluency_score,
                            'usefulness': usefulness_score,
                            'creativity': creativity_score,
                            'overall': overall_score,
                            'rationale': rationale
                        })
                    })
                    break
                except Exception:
                    continue

        result_df = pd.DataFrame(data)
        result_df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='output_test.csv')
    parser.add_argument('--output_file', type=str, default='output_test_scored.csv')
    parser.add_argument('--prompt_file_pair', type=str, default='prompts/eval_prompt_pair.txt')
    parser.add_argument('--prompt_file_single', type=str, default='prompts/eval_prompt_single.txt')
    parser.add_argument('--comp_method', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--max_try', type=int, default=3)
    args = parser.parse_args()

    evaluator = GPT4Eval(
        model_name=args.model_name,
        temperature=args.temperature,
        max_try=args.max_try,
        eval_prompt_pair=args.prompt_file_pair,
        eval_prompt_single=args.prompt_file_single
    )

    if args.comp_method == 'pair':
        evaluator.eval_pair(input_file=args.input_file, output_file=args.output_file)
    elif args.comp_method == 'single':
        evaluator.eval_single(input_file=args.input_file, output_file=args.output_file)