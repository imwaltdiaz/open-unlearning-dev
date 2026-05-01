import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10
    def __getitem__(self, idx):
        return {
            "forget": {"input_ids": [1], "attention_mask": [1], "labels": [1]},
            "retain": {"input_ids": [2], "attention_mask": [1], "labels": [2]}
        }

class DummyModel:
    def __init__(self):
        pass

import sys
sys.path.append("src")

def collate(batch):
    return {
        "forget": {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]), "labels": torch.tensor([1])},
        "retain": {"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]), "labels": torch.tensor([1])},
    }

args = TrainingArguments(output_dir="./tmp", remove_unused_columns=True)
del args.device  # to not trigger torch init maybe
trainer = Trainer(model=torch.nn.Linear(1,1), args=args, train_dataset=DummyDataset(), data_collator=collate)
try:
    trainer.get_train_dataloader()
    print("DataLoader success!")
except Exception as e:
    import traceback
    traceback.print_exc()

