# Text Preprocessing 

```python

import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

nltk.download('punkt')

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def texts_to_sequences(texts):
    # Step 1: Clean and tokenize text
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [word_tokenize(text) for text in cleaned_texts]

    # Step 2: Flatten token list and count frequency
    all_tokens = [token for tokens in tokenized_texts for token in tokens]
    token_counts = Counter(all_tokens)

    # Step 3: Create vocabulary with unique indices
    vocab = {token: idx for idx, (token, _) in enumerate(token_counts.items(), start=1)}
    vocab["<unk>"] = 0  # Unknown token

    # Step 4: Convert tokenized texts to index sequences
    sequences = [[vocab.get(token, vocab["<unk>"]) for token in tokens] for tokens in tokenized_texts]

    # Step 5: Pad sequences
    sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=vocab["<unk>"])

    return padded_sequences, vocab


def sequences_to_texts(sequences, vocab):
    index_to_word = {idx: word for word, idx in vocab.items()}
    texts = [" ".join([index_to_word[idx] for idx in sequence if idx != 0]) for sequence in sequences.data.numpy()]
    return texts

```
## Example Code
```python

texts = ["This is your sample text.", "Another example text for vocabulary creation."]

sequences, vocab = texts_to_sequences(texts)
print("Vocabulary:", vocab)
print("Sequences:", sequences)
print("Texts:", sequences_to_texts(sequences, vocab))

```
## Output
```output
Vocabulary: {'this': 1, 'is': 2, 'your': 3, 'sample': 4, 'text': 5, 'another': 6, 'example': 7, 'for': 8, 'vocabulary': 9, 'creation': 10, '<unk>': 0}
Sequences: tensor([[ 1,  2,  3,  4,  5,  0],
        [ 6,  7,  5,  8,  9, 10]])
Texts: ['this is your sample text', 'another example text for vocabulary creation']
```