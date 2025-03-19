# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re


import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_config import PeftConfig, PeftType


def get_peft_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    if state_dict is None:
        state_dict = model.state_dict()
    if model.peft_config.peft_type == PeftType.LORA:
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = model.peft_config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
    else:
        to_return = {}
        if model.peft_config.inference_mode:
            prompt_embeddings = model.prompt_encoder.embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save()
        to_return["prompt_embeddings"] = prompt_embeddings
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                to_return[key] = value
    return to_return


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight



def _find_and_replace(model):
    is_target_modules_in_base_model = False
    kwargs = {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_nums": 1,
        "blc_alpha": 0.0,
        "blc_weight": 0.0,
        "fan_in_fan_out": False,
        "merge_weights": False,
    }
    target_modules = ['in_proj']
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        if isinstance(target_modules, str):
            target_module_found = re.fullmatch(target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in target_modules)
        if target_module_found: # here
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(model, key)
            bias = target.bias is not None

            if isinstance(target, torch.nn.Linear):
                new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)

            _replace_module(parent, target_name, new_module, target)
    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
    return model

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if old_module.bias is not None:
        new_module.bias = old_module.bias
    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            module.to(old_module.weight.device)



# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.lora_num = lora_nums
        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight
        self.task_types = 't2i'
        
        self.fan_in_fan_out = fan_in_fan_out
        # self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
        for i in range(self.lora_num):
            setattr(self, f"mmu_lora_A{i}", nn.Linear(in_features, r, bias=False))
            setattr(self, f"mmu_lora_B{i}", nn.Linear(r, out_features, bias=False))
            setattr(self, f"t2i_lora_A{i}", nn.Linear(in_features, r, bias=False))
            setattr(self, f"t2i_lora_B{i}", nn.Linear(r, out_features, bias=False))
        self.scaling = self.lora_alpha / self.r
        self.weight.requires_grad = False
        self.reset_parameters()


    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "mmu_lora_A0"):
            for i in range(self.lora_num):
                nn.init.kaiming_uniform_(getattr(self, f"mmu_lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"mmu_lora_B{i}").weight)
                nn.init.kaiming_uniform_(getattr(self, f"t2i_lora_A{i}").weight, a=math.sqrt(5))
                nn.init.zeros_(getattr(self, f"t2i_lora_B{i}").weight)


    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        for i in range(self.lora_num):
            getattr(self, f"mmu_lora_A{i}").train(mode)
            getattr(self, f"mmu_lora_B{i}").train(mode)
            getattr(self, f"t2i_lora_A{i}").train(mode)
            getattr(self, f"t2i_lora_B{i}").train(mode)


    def eval(self):
        nn.Linear.eval(self)
        for i in range(self.lora_num):
            getattr(self, f"mmu_lora_A{i}").eval()
            getattr(self, f"mmu_lora_B{i}").eval()
            getattr(self, f"t2i_lora_A{i}").eval()
            getattr(self, f"t2i_lora_B{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(self, x: torch.Tensor):

        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.task_types == 't2i':
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            for i in range(self.lora_num):
                result = result + getattr(self, f"t2i_lora_B{i}")(getattr(self, f"t2i_lora_A{i}")(self.lora_dropout(x))) * self.scaling
        elif self.task_types == 'mmu':
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            for i in range(self.lora_num):
                result = result + getattr(self, f"mmu_lora_B{i}")(getattr(self, f"mmu_lora_A{i}")(self.lora_dropout(x))) * self.scaling
            
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
        return result
