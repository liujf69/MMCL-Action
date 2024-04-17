import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.blip_vqa import blip_vqa
from torchvision.transforms.functional import InterpolationMode

def load_demo_image(image_name, image_size, device):
    raw_image = Image.open(image_name).convert('RGB')   
    w,h = raw_image.size    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 480
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    sample_path = './ntu120.txt'
    sample_name = np.loadtxt(sample_path, dtype = str)
    save_path = './blip_output/object/'
    data_path = './img/'
    for _, name in enumerate(sample_name):
        print("Processing " + name)
        if int(name[-3: ]) > 60: 
            image_name = data_path + 'ntu120/' + name + '.jpg'
        else:
            image_name = data_path + 'ntu60/' + name + '.jpg'
        image = load_demo_image(image_name, image_size=image_size, device=device)    
        question = 'whether the people are holding objects?'
        with torch.no_grad():
            output = model(image, question, train = False, inference = 'generate') 
            np.save(save_path + name + '.npy', output.cpu())
    
    print("All done!")