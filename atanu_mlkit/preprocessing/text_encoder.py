import torch
from torch.nn.utils.rnn import pad_sequence

from collections import Counter

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
