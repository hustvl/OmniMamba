import torch
from transformers import Trainer
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.utilities import CombinedLoader
from util.coco_dataset import MSCOCOFeatureDataset, coco_collate_fn
from util.llava_instruct_data import get_dataset_and_collator



class Stage2Trainer(Trainer):
    def __init__(self, model, args, configs=None):
        super().__init__(model, args)
        self.configs = configs
        self.loss_t2i = torch.tensor(0.0)
        self.loss_mmu = torch.tensor(0.0)


    def get_model(self, model):
        return model.module if hasattr(model, 'module') else model

    def get_eval_dataloader(self):
        model = self.get_model(self.model)
        # only for t2i task
        if self.configs.t2i_task:
            train_dataset_t2i_path = self.configs.dataset
            eval_dataset_t2i = MSCOCOFeatureDataset(train_dataset_t2i_path.replace('train', 'val'), model.llm_backbone.uni_prompting, sample_num=6)
            if self.accelerator.num_processes > 1:
                sampler = DistributedSampler(eval_dataset_t2i,
                                            num_replicas=self.accelerator.num_processes,
                                            rank=self.accelerator.process_index,
                                            shuffle=False,
                                            drop_last=True,
                                            )
                shuffle = False
            else:
                sampler = RandomSampler(eval_dataset_t2i)
                shuffle = False
            collate_fn = coco_collate_fn(model.llm_backbone.uni_prompting.text_tokenizer.pad_token_id)
            eval_dataloader_t2i = DataLoader(eval_dataset_t2i, batch_size=25,pin_memory=True,
                                            sampler=sampler, collate_fn=collate_fn, persistent_workers=True,
                                            shuffle=shuffle, num_workers=16)
        return self.accelerator.prepare(eval_dataloader_t2i)


    def get_train_dataloader(self):
        model = self.get_model(self.model)
        if self.configs.t2i_task:
            train_dataset_t2i = MSCOCOFeatureDataset(self.configs.dataset, model.llm_backbone.uni_prompting)
            if self.accelerator.num_processes > 1:
                sampler = DistributedSampler(train_dataset_t2i,
                                            num_replicas=self.accelerator.num_processes,
                                            rank=self.accelerator.process_index,
                                            shuffle=True,
                                            drop_last=True,
                                            )
                shuffle = False
            else:
                sampler = RandomSampler(train_dataset_t2i)
                shuffle = False
            collate_fn = coco_collate_fn(model.llm_backbone.uni_prompting.text_tokenizer.pad_token_id)
            train_dataloader_t2i = DataLoader(train_dataset_t2i, batch_size=self.configs.batch_size_t2i,pin_memory=True,
                                            sampler=sampler, collate_fn=collate_fn, persistent_workers=True,
                                            shuffle=shuffle, num_workers=16)
            print(f"t2i dataloader length: {len(train_dataloader_t2i)}")
        if self.configs.mmu_task:
            # llava dataset
            train_dataset_mmu, collator_mmu = get_dataset_and_collator(
                self.configs.stage,
                'None',
                model.image_transform,
                model.llm_backbone.uni_prompting.text_tokenizer,
                prompt_builder_fn=model.llm_backbone.prompt_builder_fn,
                default_image_resolution=model.vision_backbone.default_image_resolution,
                padding_side=model.llm_backbone.uni_prompting.text_tokenizer.padding_side,
                max_length=449,
                eot_id=model.llm_backbone.uni_prompting.sptids_dict['<|eot|>']
            )
            if self.accelerator.num_processes > 1:
                datasampler_mmu = DistributedSampler(train_dataset_mmu,
                                            num_replicas=self.accelerator.num_processes,
                                            rank=self.accelerator.process_index,
                                            shuffle=True,
                                            drop_last=True,
                                            )
                shuffle = False
            else:
                datasampler_mmu = RandomSampler(train_dataset_mmu)
                shuffle = False
            train_dataloader_mmu = DataLoader(
                train_dataset_mmu,
                batch_size=self.configs.batch_size_mmu,
                num_workers=16,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
                collate_fn=collator_mmu,
                sampler=datasampler_mmu,
            )
            print(f"mmu dataloader length: {len(train_dataloader_mmu)}")

        iterables = {
            key: value for key, value in {
                "t2i_flow": train_dataloader_t2i if self.configs.t2i_task else None,
                "mmu_flow": train_dataloader_mmu if self.configs.mmu_task else None,
            }.items() if value is not None
        }
        combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
        iter(combined_dataloader)
        print(f"Combined dataloader length: {len(combined_dataloader)}")
        return self.accelerator.prepare(combined_dataloader)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        inputs = inputs[0]
        if self.configs.t2i_task:
            self.loss_t2i = model(inputs, task='t2i')
        if self.configs.mmu_task:
            self.loss_mmu = model(inputs,task='mmu')
        
        loss = self.loss_t2i + self.loss_mmu
        self.loss_dict = {
            "loss_t2i": self.loss_t2i.item(),
            "loss_mmu": self.loss_mmu.item(),
        }

        return (loss, None) if return_outputs else loss
    
    def log(self, logs: dict):
        if hasattr(self, "loss_dict"):
            logs.update(self.loss_dict)
        super().log(logs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        epoch = int(self.state.epoch) if self.state.epoch else 0
        if epoch == 100 and self.configs.t2i_task:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            self.model.eval()
            self.accelerator.wait_for_everyone()
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                with torch.no_grad():
                    loss = self.compute_loss(self.model, batch)
                    loss_accumulator += loss.item()

        avg_loss = loss_accumulator / len(eval_dataloader)
        metrics = {f"{metric_key_prefix}_loss": avg_loss}
        self.log(metrics)

        return metrics

    def get_decay_parameter_names(self, model):
        if hasattr(model, 'mamba'):
            param_dict = {pn: p for pn, p in model.mamba.named_parameters()}    
        else:
            param_dict = {pn: p for pn, p in model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_parameters = [n for n, p in param_dict.items() if p.dim() >= 2]
        return decay_parameters