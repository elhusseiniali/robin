from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import re


def get_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    ).to("cuda")
    return tokenizer, model


def prepare_prompt(message):
    return f"<s>[INST] {message.strip()} [/INST]"


def run_model(model_id, prompt):
    tokenizer, model = get_model(model_id)
    prompt = prepare_prompt(prompt)

    tokenized_prompt = tokenizer(
        prompt, return_tensors='pt',
        add_special_tokens=False
        )
    input_ids = tokenized_prompt['input_ids'].to('cuda')

    output = model.generate(
        input_ids,
        max_new_tokens=200,
        )
    output = output[0].to('cpu')
    filling = tokenizer.decode(
        output[input_ids.shape[1]:],
        skip_special_tokens=True
        )

    return filling


def process_output(output):
    pattern = re.compile(r'\[ANS\](.*?)\[/ANS\]')

    matches = pattern.findall(output)
    extracted_text = matches[0].strip() if matches else None
    return extracted_text



def pipeline(prompt):
    model_id = "codellama/CodeLlama-7b-hf"
    output = run_model(model_id, prompt)

    return process_output(output)



if __name__ == '__main__': 
    model_id = "codellama/CodeLlama-7b-hf"


    prompt = '''def remove_non_ascii(s: str) -> str:
        """ <FILL_ME>
        return result
    '''

    pipeline(model_id=model_id, prompt=prompt)
