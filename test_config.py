import sys
import os
from omegaconf import OmegaConf

sys.path.append("src")
from data import get_data
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Simulate Attempt 1
    new_cfg_data = OmegaConf.create({
        "anchor": "forget",
        "forget": {
            "my_forget": {
                "handler": "QADataset",
                "args": {
                    "hf_args": {
                        "path": "json",
                        "data_files": "data/historia/sequential_splits/forget_batch_1.jsonl",
                        "split": "train"
                    },
                    "question_key": "question",
                    "answer_key": "answer",
                    "max_length": 512
                }
            }
        },
        "retain": {
            "my_retain": {
                "handler": "QADataset",
                "args": {
                    "hf_args": {
                        "path": "json",
                        "data_files": "data/historia/processed/retain.jsonl",
                        "split": "train"
                    },
                    "question_key": "question",
                    "answer_key": "answer",
                    "max_length": 512
                }
            }
        }
    })
    
    try:
        template_args = {
            "apply_chat_template": True
        }
        data = get_data(new_cfg_data, mode="unlearn", tokenizer=tokenizer, template_args=template_args)
        train_ds = data["train"]
        print("Type train: ", type(train_ds))
        print("Type forget:", type(train_ds.forget))
        print("Type retain:", type(train_ds.retain))
        print("Length forget: ", len(train_ds.forget))
        print("Length retain: ", len(train_ds.retain))
        
        # Test __getitem__
        item = train_ds[0]
        print("Success fetching index 0. Keys:", item.keys())
        
        # Let's test DataLoader / Collator!
        from data.collators import DataCollatorForSupervisedDataset
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        batch = [train_ds[0], train_ds[1]]
        collated = collator(batch)
        print("Collator success! Keys:", collated.keys())
        print(collated)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
