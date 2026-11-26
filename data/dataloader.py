from torch.utils.data import DataLoader
from functools import partial
from .dataset import SMSDataset, collate_fn

def create_dataloader(data, tokenizer, batch_size=32, shuffle=False):
    dataset = SMSDataset(data)
    # Partial function to pass the tokenizer to the collate_fn
    # As dataloader does not allow passing additional arguments to collate_fn.

"""
1. Default Collate (Fixed-Size Data)
When all samples have the exact same shape, 
default_collate simply stacks them together along a new batch dimension.

2. Custom Collate (Variable-Size Data)
When samples have different lengths (like sentences or audio), 
default_collate crashes because you cannot stack ragged tensors. 
A custom collate_fn pads shorter items so they align into a single rectangle.
"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )