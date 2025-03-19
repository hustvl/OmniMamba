import copy
import json
import os
from functools import partial

import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, GPTNeoXTokenizerFast
from typing import Dict, List, Tuple, Type
from models.cobra.backbones.llm.prompting import PromptBuilder
from models.cobra.backbones.vision import ImageTransform
from pathlib import Path
from models.cobra.data_utils import PaddedCollatorForLanguageModeling

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
# conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]



class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 381,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)
        

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        max_length = min(self.max_length, input_ids.size(0))
        input_ids_padding = torch.fill_(torch.zeros(self.max_length, dtype=torch.long), self.pad_token_id)
        labels_padding = torch.fill_(torch.zeros(self.max_length, dtype=torch.long), IGNORE_INDEX)
        input_ids_padding[:max_length] = input_ids[:max_length]
        labels_padding[:max_length] = labels[:max_length]

        # For tokenizers that have the <BOS> token: 
        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
        if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        image_path = str(self.image_dir / image_path)
        try:
            pixel_values = self.image_transform(Image.open(image_path).convert("RGB"))
            return dict(pixel_values=pixel_values, input_ids=input_ids_padding, labels=labels_padding)
        except:
            try:
                image_path = image_path.replace('jpg','gif')
                pixel_values = self.image_transform(Image.open(image_path).convert("RGB"))
                return dict(pixel_values=pixel_values, input_ids=input_ids_padding, labels=labels_padding)
            except:
                print("Read image error. Use dummy data.", image_path)
                return dict(pixel_values=None, input_ids=input_ids_padding, labels=labels_padding)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        max_length: int = 381,
        eot_id: int = 0,
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"
        self.max_length = max_length
        self.eot_id = eot_id
        self.pad_token_id = tokenizer.pad_token_id

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)
        self.examples = [example for example in self.examples if "image" in example]
        # print(f"Loaded {len(self.examples)} examples for finetuning.")

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="cobra"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            if isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                pass
            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # Check if adding this turn will exceed max length
            if len(input_ids) + len(turn_input_ids) > self.max_length-1:
                # If so, break out of the loop without adding this turn
                break

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        # Handle Truncation (if necessary)

        input_ids = torch.cat((input_ids,(torch.ones(1) * self.eot_id)))
        labels = torch.cat((labels,(torch.ones(1) * IGNORE_INDEX).long()))

        max_length = min(self.max_length, input_ids.size(0))
        input_ids_padding = torch.fill_(torch.zeros(self.max_length, dtype=torch.long), self.pad_token_id)
        labels_padding = torch.fill_(torch.zeros(self.max_length, dtype=torch.long), IGNORE_INDEX)
        input_ids_padding[:max_length] = input_ids[:max_length]
        labels_padding[:max_length] = labels[:max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
            if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                labels[0] = IGNORE_INDEX
            image_path = str(self.image_dir / image_path)

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            try:
                pixel_values = self.image_transform(Image.open(image_path).convert("RGB"))
                return dict(pixel_values=pixel_values, input_ids=input_ids_padding, labels=labels_padding)
            except:
                try:
                    image_path = image_path.replace('jpg','gif')
                    pixel_values = self.image_transform(Image.open(image_path).convert("RGB"))
                    return dict(pixel_values=pixel_values, input_ids=input_ids_padding, labels=labels_padding)
                except:
                    print("Read image error. Use dummy data.", image_path)
                    return dict(pixel_values=None, input_ids=input_ids_padding, labels=labels_padding)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            print("no image data.")
            return dict(pixel_values=None, input_ids=input_ids_padding, labels=labels_padding)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)




DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    max_length: int = 381,
    eot_id: int=0,
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = Path("datasets/")
    collator = PaddedCollatorForLanguageModeling(
        max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        align_stage_components: Tuple[Path, Path] = (
            Path("llava/share-captioner_coco_lcs_676k_1121.json"),
            Path("llava/"),
        )
        annotation_json, image_dir = align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer, max_length=max_length,
        )
        return dataset, collator

    elif stage == "finetune":
        finetune_stage_components: Tuple[Path, Path] = (
            # Path("llava/llava_v1_5_mix665k.json"),
            Path("llava/sharegpt4v_llava_v1_5_lvis4v_lrv_mix1231k.json"),
            Path("llava/"),
        )
        annotation_json, image_dir = finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            eot_id=eot_id,
            max_length=max_length,
        )
        return dataset, collator
    else:
        raise ValueError(f"Stage `{stage}` is not supported!")


