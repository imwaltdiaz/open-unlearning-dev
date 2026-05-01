import sys
sys.path.append("src")
from data.unlearn import ForgetRetainDataset
from data.qa import QADataset
from data.collators import DataCollatorForSupervisedDataset
from transformers import AutoTokenizer
import torch

# Create mock dataset
class MockDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3])
        }

# Create ForgetRetainDataset-like structure
forget_ds = MockDataset()
retain_ds = MockDataset()
composite_ds = ForgetRetainDataset(forget=forget_ds, retain=retain_ds, anchor="forget")

# Create tokenizer and collator
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# Get 2 samples from composite dataset
samples = [composite_ds[0], composite_ds[1]]
print("=" * 80)
print("SAMPLES FROM DATASET:")
for i, sample in enumerate(samples):
    print(f"  Sample {i}: keys={sample.keys()}")
    for k, v in sample.items():
        if isinstance(v, dict):
            print(f"    {k}: {v.keys()}")
        else:
            print(f"    {k}: {type(v)}")

# Call collator
print("\n" + "=" * 80)
print("CALLING COLLATOR...")
batch = collator(samples)

print("\n" + "=" * 80)
print("BATCH FROM COLLATOR:")
print(f"Top-level keys: {batch.keys()}")
for key in batch:
    if isinstance(batch[key], dict):
        print(f"  {key}: {batch[key].keys()}")
    else:
        print(f"  {key}: shape={batch[key].shape if hasattr(batch[key], 'shape') else 'N/A'}")
