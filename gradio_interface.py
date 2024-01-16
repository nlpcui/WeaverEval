import json
import logging
import random
import uuid

import gradio
import gradio as gr
import pandas as pd
import argparse
import hashlib


def find_between(string, prefix, suffix):
    prefix_idx = string.find(prefix) + len(prefix)
    suffix_idx = string.find(suffix)
    return string[prefix_idx: suffix_idx]


def load_next(batch_id, domain, task, instruction, output_1, output_2, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, example_info):
    if batch_id not in annotation_batches:
        gr.Warning('任务编号错误，请重新输入！')
        return domain, task, output_1, output_2, instruction, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, example_info
    else:
        finished_cnt = sum([res is not None for res in result[batch_id]])
        if finished_cnt >= len(annotation_batches[batch_id]):
            gradio.Info('您已经完成所有标注任务，感谢您的参与！')
            return '', '', '', '', '', '', '', '', '', '', ''

        next_example_id = finished_cnt
        next_example_info = '下一组 （{}/{}）'.format(next_example_id+1, len(annotation_batches[batch_id]))

        domain_next = annotation_batches[batch_id][next_example_id]['domain']
        task_next = annotation_batches[batch_id][next_example_id]['task']
        instruction_next = annotation_batches[batch_id][next_example_id]['instruction']
        output_1_next = annotation_batches[batch_id][next_example_id]['model_A_output']
        output_2_next = annotation_batches[batch_id][next_example_id]['model_B_output']

        return domain_next, task_next, instruction_next, output_1_next, output_2_next, '', '', '', '', '', next_example_info


def submit(batch_id, domain, task, output_1, output_2, instruction, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, example_info):
    example_id = find_between(example_info, '（', '）')
    n1, n2 = example_id.split('/')
    example_id, example_cnt = int(n1), int(n2)

    if batch_id not in annotation_batches:
        gradio.Warning('请输入正确的任务编号！')
        return domain, task, output_1, output_2, instruction, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, example_info

    if example_id == 0:
        gradio.Warning('请先输入任务编号！')
        return domain, task, output_1, output_2, instruction, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, example_info

    if not all([usefulness_radio, relevance_radio, fluency_radio, overall_radio]):
        gradio.Warning('请完成所有标注！')
        return domain, task, output_1, output_2, instruction, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, example_info

    # save data
    result[batch_id][example_id-1] = [relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio]
    cache_fp.write(json.dumps({
        'batch_id': batch_id,
        'example_id': example_id-1,
        'usefulness': usefulness_radio,
        'relevance': relevance_radio,
        'fluency_radio': fluency_radio,
        'creativity_radio': creativity_radio,
        'overall_radio': overall_radio
    })+'\n')
    cache_fp.flush()

    # load next
    domain_next, task_next, instruction_next, output_1_next, output_2_next, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, next_example_info = load_next(
        batch_id, domain, task, output_1, output_2, instruction, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, example_info
    )

    return domain_next, task_next, instruction_next, output_1_next, output_2_next, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, next_example_info


def split_data(input_file, n):
    data = []
    data_frame = pd.read_csv(input_file)
    for row_id, row in data_frame.iterrows():
        data.append({
            'id': str(row['id']),
            'domain': row['domain'],
            'task': row['task'],
            'instruction': row['instruction'],
            'model_A_name': row['model_A_name'],
            'model_B_name': row['model_B_name'],
            'model_A_output': row['model_A_output'],
            'model_B_output': row['model_B_output']
        })

    batch_size = len(data) // n
    batches = {}
    for i in range(n):
        batch = []
        start = i * batch_size
        end = (i+1) * batch_size if i != n-1 else len(data)
        for j in range(start, end):
            batch.append(data[j])

        concat_ids = '#'.join([item['id'] for item in batch])
        batch_id = hashlib.md5(concat_ids.encode(encoding='UTF-8')).hexdigest()
        batches[batch_id] = batch

    return batches


def start_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Group():
                with gr.Column(scale=5):
                    id_input = gr.Textbox(label='任务编号')
                with gr.Column(scale=1):
                    confirm_button = gr.Button(value='确认')

        with gr.Row():
            domain = gr.Textbox('', label="领域")
            task = gr.Textbox('', label="任务")

        with gr.Row():
            instruction = gr.Textbox('', label='指令')

        with gr.Row():
            with gr.Column(scale=3, min_width=600):
                output_1 = gr.Textbox('', label="模型 1 输出")
            with gr.Column(scale=3, min_width=600):
                output_2 = gr.Textbox('', label='模型 2 输出')

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab('相关性'):
                    relevance_radio = gr.Radio(choices=['左边好', '右边好', '一样好'], label='', )  # info='模型输出和指令是否相关')
                with gr.Tab('评价标准'):
                    relevance_definition = gr.Textbox('内容和指令的要求是否匹配。', label='')

            with gr.Column(scale=2):
                with gr.Tab('流畅性'):
                    fluency_radio = gr.Radio(choices=['左边好', '右边好', '一样好'], label='', )  # info='文本是否流畅，连续')
                with gr.Tab('评价标准'):
                    fluency_definition = gr.Textbox('内容是否语法正确、逻辑连贯、结构合理、阅读时难以辨别是机器创作。', label='')

            with gr.Column(scale=2):
                with gr.Tab('创意性比较'):
                    creativity_radio = gr.Radio(choices=['左边好', '右边好', '一样好'], label='')  # info='模型输出是否有创意')
                with gr.Tab('评价标准'):
                    usefulness_definition = gr.Textbox('内容是否包含新颖的想法、剧情或方案。', label='')

            with gr.Column(scale=2):
                with gr.Tab('有用性比较'):
                    usefulness_radio = gr.Radio(choices=['左边好', '右边好', '一样好'], label='')  # info='模型输出是否有创意')
                with gr.Tab('评价标准'):
                    usefulness_definition = gr.Textbox('内容是否能够帮助解决用户指令中的问题和需求，需要结合任务和领域作出评价。', label='')

            with gr.Column(scale=8):
                with gr.Tab('综合质量'):
                    overall_radio = gr.Radio(choices=['左边好', '右边好', '一样好'], label='整体质量', )  # info='综合质量')
                with gr.Tab('评价标准'):
                    overall_definition = gr.Textbox('结合以上所有评价维度的综合比较。', label='')

        with gr.Row():
            submit_button = gr.Button('下一组 （{}/{}）'.format(0, 0))

        confirm_button.click(
            fn=load_next,
            inputs=[id_input, domain, task, instruction, output_1, output_2, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, submit_button],
            outputs=[domain, task, instruction, output_1, output_2, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, submit_button]
        )
        submit_button.click(
            fn=submit,
            inputs=[id_input, domain, task, instruction, output_1, output_2, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, submit_button],
            outputs=[domain, task, instruction, output_1, output_2, relevance_radio, fluency_radio, usefulness_radio, creativity_radio, overall_radio, submit_button]
        )

    demo.launch(share=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='output_test.csv')
    parser.add_argument('--num_annotator', type=int, default=4)
    parser.add_argument('--result_file', type=str, default='result.jsonl')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        # filename='log/{}qg_{}.log'.format('flan_t5_large', 'textbook')
        level=logging.INFO,
        filemode='w'
    )
    annotation_batches = split_data(input_file=args.input_file, n=args.num_annotator)  # {str: batch_id, list: batch_data}
    logging.info('batch_ids: {}'.format(annotation_batches.keys()))
    cache_fp = open(args.result_file, 'w')
    result = {batch_id: [None for i in range(len(annotation_batches[batch_id]))] for batch_id in annotation_batches}

    start_demo()

    cache_fp.close()