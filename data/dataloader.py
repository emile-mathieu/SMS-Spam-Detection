from torch.utils.data import DataLoader
from functools import partial

def create_dataloader(data, tokenizer, batch_size=32, shuffle=False):
    dataset = SMSDataset(data)
    # Partial function to pass the tokenizer to the collate_fn
    # As dataloader does not allow passing additional arguments to collate_fn.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )