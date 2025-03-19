import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from models.omnimamba import OmniMamba
import torch
import yaml
import argparse
from PIL import Image
from util import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]

parser = argparse.ArgumentParser(description="Run inference with OmniMamba model.")
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
parser.add_argument('--question', type=str, required=True, help='Question to ask about the image.')
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

image_path = args.image_path
questions = args.question


image_ori = Image.open(image_path).convert("RGB")
questions = questions.split(' *** ')
batch_size = 1
generated_texts = []
for question in questions:
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    question_input = []
    question_input.append(prompt_question.strip())
    

    input_ids = [model.llm_backbone.uni_prompting.text_tokenizer(prompt, return_tensors="pt", padding="longest").input_ids
                    for prompt in question_input]

    input_ids = torch.stack(input_ids)
    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=model.llm_backbone.uni_prompting.text_tokenizer.pad_token_id
    )
    input_ids = torch.tensor(input_ids).to('cuda').squeeze(0)
    input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * model.llm_backbone.uni_prompting.sptids_dict['<|mmu|>']).to('cuda'),
            (torch.ones(input_ids.shape[0], 1) * model.llm_backbone.uni_prompting.sptids_dict['<|soi|>']).to('cuda'),
            # place your img embedding here
            (torch.ones(input_ids.shape[0], 1) * model.llm_backbone.uni_prompting.sptids_dict['<|eoi|>']).to('cuda'),
            (torch.ones(input_ids.shape[0], 1) * model.llm_backbone.uni_prompting.sptids_dict['<|sot|>']).to('cuda'),
            input_ids,
    ], dim=1).long()


    image_transform = model.vision_backbone.image_transform
    pixel_values = image_transform(image_ori)
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to('cuda')
    elif isinstance(pixel_values, dict):
        pixel_values = {k: v[None, ...].to('cuda') for k, v in pixel_values.items()}
    else:
        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    images_feat = model.vision_backbone(pixel_values)
    images_embeddings = model.projector(images_feat)
    text_embeddings = model.llm_backbone.embed_input_ids(input_ids)

    # Full input seq
    part1 = text_embeddings[:, :2, :]
    part2 = text_embeddings[:, 2:, :]
    input_embeddings_mmu = torch.cat((part1, images_embeddings, part2), dim=1)


    generated_ids = model.llm_backbone.mamba.generate(
                            input_ids=input_ids,
                            input_embeddings=input_embeddings_mmu,
                            cond=None,
                            eos_token_id=model.llm_backbone.uni_prompting.text_tokenizer.eos_token_id,
                            max_length=2048,
                            temperature=1.0,
                            top_p=0.0,
                            top_k=1,
                            cg=True,
                            task='mmu',)
    logits = generated_ids.squeeze()
    generated_text = model.llm_backbone.uni_prompting.text_tokenizer.decode(logits)
    generated_texts.append(generated_text)
print(generated_texts)

