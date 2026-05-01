import sys
from omegaconf import OmegaConf
sys.path.append("src")
from data import get_data
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Simulate Attempt 1 WITHOUT split="train" in hf_args!
    new_cfg_data = OmegaConf.create({
        "anchor": "forget",
        "forget": {
            "my_forget": {
                "handler": "QADataset",
                "args": {
                    "hf_args": {
                        "path": "json",
                        "data_files": "data/historia/sequential_splits/forget_batch_1.jsonl",
                        # "split": "train"   <--- OMITTING THIS!
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
                        "split": "train" # Let's say this has split
                    },
                    "question_key": "question",
                    "answer_key": "answer",
                    "max_length": 512
                }
            }
        }
    })
    
    template_args = {"apply_chat_template": True}
    data = get_data(new_cfg_data, mode="unlearn", tokenizer=tokenizer, template_args=template_args)
    train_ds = data["train"]
    print("Testing index 0...")
    try:
        item = train_ds[0]
        print("Success!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
