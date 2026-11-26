# SMS Spam and Smishing Detection

A text classification project for detecting spam and phishing SMS messages using a custom Byte-Level BPE tokenizer and a Transformer encoder built in PyTorch.

The project is motivated by the relevance of mobile messaging scams, particularly in Singapore. It aims to explore an end-to-end machine learning pipeline, from dataset preparation and tokenization to model training and evaluation.

## Project Status

- [x] Dataset acquisition and exploration
- [x] Data cleaning and label encoding
- [x] Stratified train, validation and test splitting
- [x] Custom BPE tokenizer training
- [ ] Implement PyTorch Dataset and DataLoader
- [ ] Implement Transformer encoder
- [ ] Train and evaluate the model with baseline models

## Dataset License and Attribution

The project uses the [SMS Phishing Dataset for Machine Learning and Pattern Recognition](https://data.mendeley.com/datasets/f45bkkt8pr/1), published by Sandhya Mishra and Devpriya Soni on Mendeley Data on 20 June 2022, under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence.

The original dataset contains 5,971 labelled SMS messages across three classes: ham, spam and smishing. Following preprocessing, including whitespace normalisation, duplicate removal and removal of messages with conflicting labels, 5,797 samples were retained.

Only the `TEXT` and `LABEL` columns are used for the initial text-only classification experiment. The additional `URL`, `EMAIL` and `PHONE` features are excluded for now.

| Class | Label | Processed samples |
| --- | ---: | ---: |
| Ham | 0 | 4,834 |
| Spam | 1 | 435 |
| Smishing | 2 | 528 |
| **Total** | | **5,797** |

Note: The dataset is imbalanced, with legitimate messages representing approximately 83.39% of the cleaned dataset. Its publication date does not establish the collection date of every message, and it is not specifically a Singaporean or WhatsApp dataset.

## Data Preprocessing

The preprocessing pipeline is implemented in `dataset.ipynb`:

1. Download and extract the original dataset from Mendeley Data.
2. Select the message text and classification label.
3. Map the classification labels to integers.
4. Check for missing values and empty messages.
5. Strip leading and trailing whitespace and normalise consecutive whitespace.
6. Remove all occurrences of messages with conflicting labels.
7. Remove remaining duplicate messages and reset the DataFrame index.

Capitalisation, punctuation, URLs, email addresses, phone numbers and emojis are preserved in the message text.

### Cleaning Results

| Operation | Rows removed |
| --- | ---: |
| Missing values | 0 |
| Empty messages | 0 |
| Conflicting labels (34 unique messages) | 68 |
| Remaining duplicates | 106 |
| **Total** | **174** |

### Dataset Splitting

The cleaned dataset is split using scikit-learn's `train_test_split` with stratification and a random seed of `42`. Stratification preserves approximately the same class proportions across the three splits.

| Split | Proportion | Samples |
| --- | ---: | ---: |
| Training | 70% | 4,057 |
| Validation | 15% | 870 |
| Testing | 15% | 870 |

The splits are exported as separate CSV files. Exact duplicate messages are removed before splitting to reduce data leakage; near-duplicate messages may still remain.

## Tokenizer

A custom Byte-Level Byte Pair Encoding (BPE) tokenizer is trained from scratch using the Hugging Face `tokenizers` library.

| Configuration | Value |
| --- | --- |
| Model | BPE |
| Pre-tokenizer | ByteLevel |
| Target vocabulary size | 2,000 |
| Minimum merge frequency | 2 |
| Initial alphabet | All 256 byte values |
| Prefix space | Disabled |
| Normalizer | None |
| Decoder | ByteLevel |

### Special Tokens

| Token | ID | Purpose |
| --- | ---: | --- |
| `[UNK]` | 0 | Unknown token fallback |
| `[PAD]` | 1 | Batch padding |
| `[SOS]` | 2 | Start of sequence; planned classification representation |
| `[EOS]` | 3 | End of sequence |

## Model Architecture

**TODO:**

## Training

**TODO:**

## Evaluation

**TODO:**

## Getting Started

```bash
pip install pandas requests scikit-learn tokenizers jupyter
```

1. Run `notebooks/dataset.ipynb` to download and preprocess the dataset, then export the splits. 
2. Run `notebooks/tokenizer.ipynb` to train the custom tokenizer and export it as a JSON file.
3. Todo

### Intended Project Structure

```text
├── notebooks/
│   ├── dataset.ipynb
│   └── tokenizer.ipynb
├── data/
│   ├── unprocessed/
│   │   └── Dataset_5971.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── tokenizer/
│   └── tokenizer.json
├── requirements.txt
└── README.md
```
