from transformers import AutoTokenizer, AutoModelForCausalLM


def get_model(checkpoint='bigscience/bloomz-1b7'):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype="auto", device_map="cuda")
    return tokenizer, model


def run(prompt):
    tokenizer, model = get_model()
    inputs = tokenizer.encode(prompt,
                              return_tensors='pt').to('cuda')
    outputs = model.generate(inputs)

    return tokenizer.decode(outputs[0])


if __name__ == "__main__":
    prompt = "Translate to English: Je t'aime."
    print(run(prompt))
