import os
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

def main(model_path: str, imgs_path: str, prompt: str): 
    """
    Args:
        model_path: Well pretrained model path of DeepSeek-VL.
        imgs_path: The path of test images.
        prompt: The prompt content that input to the model
    """

    # specify the path to the model
    model_path = model_path
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    imgs_path = imgs_path
    answers_dict = []
    for _, img in enumerate(os.listdir(imgs_path)):
        # single image conversation example
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>" + prompt,
                "images": [imgs_path + "/" + img]
            },
            {"role": "Assistant", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        answers_dict.append({'img': img, 'answer': answer})
        # print(f"{prepare_inputs['sft_format'][0]}", answer)

    print("All Done!")

if __name__ == "__main__":  
    model_path = "your model path"
    imgs_path = "your data path"
    prompt = "whether the people are holding objects?" # try more prompt or instuction
    main(model_path = model_path, imgs_path = imgs_path, prompt = prompt)