import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(model_path: str, query: str, image_path: str):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype = torch.bfloat16,
            low_cpu_mem_usage = True,
            trust_remote_code = True
        ).to(device).eval()

    query = query
    image_path = image_path
    answers_dict = []
    for i, name in enumerate(os.listdir(image_path)):
        image = Image.open(image_path + "/" + name).convert('RGB')
        inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                            add_generation_prompt = True, tokenize = True, return_tensors = "pt",
                                            return_dict = True)  # chat mode
        inputs = inputs.to(device)
        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # print(tokenizer.decode(outputs[0]))
            answer = tokenizer.decode(outputs[0])
            answers_dict.append({'img': name, 'answer': answer})
        
    print("All Done!")

if __name__ == "__main__":
    model_path = "your model path"
    query = 'whether the people are holding objects?' # try more prompt or instuction
    image_path = "your data path"
    main(model_path = model_path, query = query, image_path = image_path)