# Text Encoder
**TextEncoder** convert text into sequences for use with RNNs, LSTMs, and GRUs.

#### TextEncoder

```python
from collections.abc import Sequence
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

class TextEncoder:
    def __init__(self, specials=["<unk>", "<pad>"], vocab=None,  preprocessor=None):
        """Initialize the TextProcessor with optional special tokens."""
        self.specials = specials
        self.vocab = vocab
        self.idx_to_token = None
        self.preprocessor = preprocessor

    def build_vocab(self, tokenized_texts):
        """Builds a vocabulary from tokenized texts."""
        all_tokens = [token for tokens in tokenized_texts for token in tokens]
        token_counts = Counter(all_tokens)
        self.vocab = {token: idx for idx, token in enumerate(self.specials + list(token_counts.keys()))}
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

    def tokenize(self, text):
        """Tokenizes a text string."""
        if self.preprocessor is None:
            return text.lower().split()
        else:
            return self.preprocessor(text).split()

    def text_to_sequence(self, tokenized_text, unk_token="<unk>"):
        """Converts tokenized text to a sequence of indices based on the vocabulary."""
        return [self.vocab.get(token, self.vocab[unk_token]) for token in tokenized_text]

    def fit(self, texts, pad_token="<pad>"):
        """Converts a list of texts to padded sequences of indices."""
        tokenized_texts = [self.tokenize(text) for text in texts]

        # Build vocabulary if it hasn't been built yet
        if self.vocab is None:
            self.build_vocab(tokenized_texts)

        # Convert tokenized texts to sequences
        sequences = [torch.tensor(self.text_to_sequence(tokens), dtype=torch.long) for tokens in tokenized_texts]
        
        # Pad sequences
        pad_idx = self.vocab.get(pad_token, 0)  # Use 0 if pad_token not in vocab
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)

        return padded_sequences

    def transform(self, texts, pad_token="<pad>"):
        """Converts new texts to sequences using the existing vocabulary."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Please convert some texts first.")

        tokenized_texts = [self.tokenize(text) for text in texts]
        sequences = [torch.tensor(self.text_to_sequence(tokens), dtype=torch.long) for tokens in tokenized_texts]
        pad_idx = self.vocab.get(pad_token, 0)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)

        return padded_sequences

    def sequence_to_text(self, sequence, pad_token="<pad>"):
        """Converts a sequence of indices back to text."""
        return ' '.join([self.idx_to_token[idx] for idx in sequence if idx in self.idx_to_token and self.idx_to_token[idx] != pad_token])

    def inverse_transform(self, sequences, pad_token="<pad>"):
        """Converts a sequences of indices back to text using the index-to-token mapping."""
        return [self.sequence_to_text(sequence) for sequence in sequences]


```
#### Basic text pre-processing
```python
import re
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text
```
#### Convert Text to Sequences
```python
texts = ['I Love my Dog', 'You love Cat 7']

text_encoder = TextEncoder(preprocessor=clean_text)

padded_sequences = text_encoder.fit(texts)
print("Vocabulary:", text_encoder.vocab)
print("Padded Sequences for Initial Texts:", padded_sequences)
```
output
```bash
Vocabulary: {'<unk>': 0, '<pad>': 1, 'i': 2, 'love': 3, 'my': 4, 'dog': 5, 'you': 6, 'cat': 7}
Padded Sequences for Initial Texts: tensor([[2, 3, 4, 5],
        [6, 3, 7, 1]])
```
#### Convert Text to Sequences with same vocab 
```python
texts = ['I love my Cat', 'You love Pubg 8']
padded_sequences = text_encoder.transform(texts)
print("Padded Sequences for Initial Texts:", padded_sequences)
```
output
```bash
Padded Sequences for Initial Texts: tensor([[2, 3, 4, 7],
        [6, 3, 0, 1]])
```
#### Convert Sequences to Text
```python
sequences = [[2, 3, 6]]
sequences_text = text_encoder.inverse_transform(sequences)
print("Sequence to Text:", sequences_text)
```
output
```bash
Sequence to Text: ['i love you']
```






