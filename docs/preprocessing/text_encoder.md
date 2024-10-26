# Text Encoder
**TextEncoder** convert text into sequences for use with RNNs, LSTMs, and GRUs.

#### TextEncoder

```python
from atanu_mlkit.preprocessing import TextEncoder
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
text_encoder = TextEncoder(preprocessor=clean_text)

texts = ['I Love my Dog', 'You love Cat 7']

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