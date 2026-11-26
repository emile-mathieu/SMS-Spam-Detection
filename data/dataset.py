import torch 
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class SMSDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.text = data['TEXT'].tolist()
        self.labels = data['LABEL'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'label': self.labels[idx]
        }

def collate_fn(batch, tokenizer: Tokenizer):
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]'), pad_token='[PAD]')

    """batch = [
    {"text": "Hello!", "label": 0},
    {"text": "You won a prize!", "label": 1},
    {"text": "Call me later", "label": 0}
    ]
    """
    
    # Create a list of texts and associated labels from the batch
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]

    # Tokenize the texts
    encodings = tokenizer.encode_batch(texts)

    return {
        'input_ids': torch.tensor([e.ids for e in encodings], dtype=torch.long),
        'attention_mask': torch.tensor([e.attention_mask for e in encodings], dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }


