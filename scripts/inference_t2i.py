import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from models.omnimamba import OmniMamba
from torchvision import utils as vutils
from matplotlib import pyplot as plt
import torch
import yaml
import argparse

parser = argparse.ArgumentParser(description="Run inference with OmniMamba model.")
parser.add_argument('--prompt', type=str, required=True)
args = parser.parse_args()

with open("config/config_stage2.yaml", "r") as file:
    config = yaml.safe_load(file)["train"]
config['vq_model'] = "VQ-f16"

config = argparse.Namespace(**config)
model = OmniMamba(config, stage='inference')
state_dict = torch.load("ckpts/OmniMamba-1.3b.pth", map_location="cpu")
model_state_dict = model.state_dict()
model.load_state_dict(state_dict, strict=False)
model.eval()
model.to("cuda")


prompt = args.prompt 
max_token_len = 68
text_ids = model.llm_backbone.uni_prompting.text_tokenizer(prompt).input_ids
text_ids = torch.tensor(text_ids)
caption_ids_padding = torch.fill_(torch.zeros(max_token_len, dtype=torch.long), model.llm_backbone.uni_prompting.text_tokenizer.pad_token_id)
feat_len = text_ids.size(0)
feat_len = min(feat_len, max_token_len)
caption_ids_padding[-feat_len:] = text_ids[:feat_len]

t2i_prompt_text = torch.cat([
    model.llm_backbone.uni_prompting.sptids_dict['<|t2i|>'],
    model.llm_backbone.uni_prompting.sptids_dict['<|sot|>'],
    model.llm_backbone.uni_prompting.sptids_dict['<|eot|>'],
    model.llm_backbone.uni_prompting.sptids_dict['<|soi|>'],
])

text_ids = torch.cat([t2i_prompt_text[:2], caption_ids_padding, t2i_prompt_text[2:]]).to('cuda')
imgs = model.t2i_generate(text_ids=text_ids.unsqueeze(0), temperature=1.0, top_p=0.0, top_k=1, fast=True)
images = vutils.make_grid(imgs, nrow=imgs.shape[0]//2, normalize=True, value_range =(-1, 1))
images = images.detach().cpu().permute(1, 2, 0).numpy()

fig = plt.figure(dpi=300)
plt.axis('off')
plt.imsave("generated_image.jpg", images)

