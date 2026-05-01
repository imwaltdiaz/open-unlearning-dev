import torch
import transformers
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        return_dct = {}
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            if self.index:
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct


class DataCollatorForSupervisedDatasetwithIndex(object):
    """Collate examples for supervised fine-tuning with index tracking."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = "index",
    ):
        self.base_collator = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            padding_side=padding_side,
            index=index,
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return self.base_collator(instances)


class DataCollatorForUnlearning(object):
    """Collate examples for unlearning methods (e.g., WGA, GradDiff, PDU).
    
    Preserves the nested structure {"forget": {...}, "retain": {...}} in the batch.
    Each subset (forget/retain) is processed independently to handle sequences of different lengths.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.base_collator = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            padding_side=padding_side,
            index=None,
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch while preserving forget/retain structure.
        
        Args:
            instances: List of dicts with structure {"forget": {...}, "retain": {...}}
                      Each inner dict has "input_ids", "attention_mask", "labels"
        
        Returns:
            Dict with structure {"forget": {...}, "retain": {...}} where each subset
            has padded tensors for "input_ids", "attention_mask", "labels"
        """
        assert isinstance(instances[0], dict), "Expected list of dicts"
        
        # Check if this is a forget/retain structure
        if "forget" in instances[0] and "retain" in instances[0]:
            # Extract forget and retain subsets
            forget_instances = [inst["forget"] for inst in instances]
            retain_instances = [inst["retain"] for inst in instances]
            
            # Process each subset independently using the base collator
            forget_batch = self.base_collator(forget_instances)
            retain_batch = self.base_collator(retain_instances)
            
            # Return with structure preserved
            return {
                "forget": forget_batch,
                "retain": retain_batch,
            }
        else:
            # Fall back to standard processing if not unlearning structure
            return self.base_collator(instances)
