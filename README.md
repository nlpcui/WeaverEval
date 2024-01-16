# WeaverEval

## GPT-4 evaluation
```python auto_eval --input_file=<filename> --output_file=<filename> --prompt_file=prompts/eval_prompt.txt```

## Interface

```python gradio_interface.py --input_file=<filename> --batch_ids_file=<filename> --num_annotator=4 --result_file=<filename>``` \
```input_file```: CSV file with the following fields: id, domain, task, instruction, model_A_name,  model_A_output, model_B_name, model_B_output \
```batch_ids_file```: automatically generated after starting the demo. Assign one to each annotator and input it in the Interface to load data. \
