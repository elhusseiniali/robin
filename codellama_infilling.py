from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


def get_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    ).to("cuda")
    return tokenizer, model


def run_model(model_id, prompt):
    tokenizer, model = get_model(model_id)
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')
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


def process_filling(prompt, filling):
    return prompt.replace('<FILL_ME>', filling)



def pipeline(prompt):
    model_id = "codellama/CodeLlama-7b-hf"
    filling = run_model(model_id, prompt)
    filling = process_filling(prompt, filling)
    # print(filling)
    return filling



if __name__ == '__main__': 
    model_id = "codellama/CodeLlama-7b-hf"


    prompt = '''def remove_non_ascii(s: str) -> str:
        """ <FILL_ME>
        return result
    '''

    pipeline(model_id=model_id, prompt=prompt)
