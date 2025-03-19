from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import einops
import linecache
import random
import json
import torch
from torch.nn.utils.rnn import pad_sequence


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target
    

class coco_collate_fn():
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, examples):
        inputs = torch.stack([example[0] for example in examples])
        caption_ids = torch.stack([example[1] for example in examples])

        return {"inputs":inputs, "caption_ids":caption_ids}

class coco_eval_collate_fn():
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, examples):
        caption_ids = torch.stack([example[0] for example in examples])
        caption = [example[1] for example in examples]
        return {"caption_ids":caption_ids, 'caption':caption}

class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, data_path, uni_prompting, sample_num=None):
        self.jsonl_file = data_path
        self.drop_cond_prob = 0
        self.null_prompt = 'A picture'
        self.uni_prompting = uni_prompting
        self.max_token_len = 68
        self.pad_token_id = self.uni_prompting.text_tokenizer.pad_token_id
        self.t2i_prompt_text = torch.cat([
            self.uni_prompting.sptids_dict['<|t2i|>'],
            self.uni_prompting.sptids_dict['<|sot|>'],
            self.uni_prompting.sptids_dict['<|eot|>'],
            self.uni_prompting.sptids_dict['<|soi|>'],
            self.uni_prompting.sptids_dict['<|eoi|>']
        ])     
        if sample_num is not None:
            self.eval = True
        else:
            self.eval = False
        self.lines = self._load_and_shuffle(sample_num)
        self.num_lines = len(self.lines)
        print("Number of data:", self.num_lines)

    def _load_and_shuffle(self, sample_num):
        with open(self.jsonl_file, 'r') as f:
            lines = f.readlines()
        random.shuffle(lines)
        if sample_num is not None:
            lines = lines[:sample_num]
        return lines

    def __len__(self):
        return self.num_lines

    def __getitem__(self, index):
        line = self.lines[index]
        data = json.loads(line)
        captions = data['captions']
        probs = torch.rand(1)
        if probs.item() < self.drop_cond_prob and not self.eval:
            caption = self.null_prompt
        else:
            k = random.randint(0, len(captions) - 1)
            caption = captions[k]
        
        caption_ids_padding = torch.fill_(torch.zeros(self.max_token_len, dtype=torch.long), self.pad_token_id)
        caption_ids = self.uni_prompting.text_tokenizer(caption).input_ids
        caption_ids = torch.tensor(caption_ids)
        feat_len = caption_ids.size(0)
        feat_len = min(feat_len, self.max_token_len)
        caption_ids_padding[-feat_len:] = caption_ids[:feat_len]
        caption_ids_padding = torch.cat([self.t2i_prompt_text[:2], caption_ids_padding, self.t2i_prompt_text[2:]])
        if self.eval:
            return caption_ids_padding[:-1], caption # no eoi
        else:
            tokens = data['tokens']
            return torch.tensor(tokens), caption_ids_padding

