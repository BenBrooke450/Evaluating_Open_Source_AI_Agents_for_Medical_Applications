


class LLM_Tiny_Llama():
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("./tinyllama")
        model = AutoModelForCausalLM.from_pretrained("./tinyllama")

    def prompt(self, prompt):

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))