import torch
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel

from DeepSC_torch.dataloader import Dataset


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

class tokenizer():
    def __init__(self, config):
        # Place-holders
        self.token_transform = {}
        self.vocab_transform = {}

        # Token transform: a sentence is transformed into a list of characters
        # ex) 'a b c' => ['a', 'b', 'c']
        self.token_transform = get_tokenizer('spacy', language='en_core_web_sm')

        # Vocab transform transforms each character into the corresponding index.
        # ex) ['Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.']
        # => [21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4]

        # Training data Iterator
        train_iter = iter(Dataset(config, 'train'))
        if os.path.isfile("vocab"):
            self.vocab_transform = torch.load("vocab")
        else:
            print("Start creating vocabulary !")
            # Create torchtext's Vocab object
            self.vocab_transform = build_vocab_from_iterator(self.yield_tokens(train_iter),
                                                            min_freq=1,
                                                            specials=special_symbols,
                                                            special_first=True)
            print("Creating vocabulary finished !")
            torch.save(self.vocab_transform, "vocab")

        # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
        self.vocab_transform.set_default_index(UNK_IDX)

    # helper function to yield list of tokens
    def yield_tokens(self, data_iter: Iterable) -> List[str]:
        for data_sample in data_iter:
            yield self.token_transform(data_sample)

    # function to collate data samples into batch tensors
    def collate_fn(self, batch):
        src_batch = []
        for src_sample in batch:
            src_batch.append(self.text_transform()(src_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)

        return src_batch

    def text_transform(self):
        # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
        text_transform = sequential_transforms(self.token_transform,  # Tokenization
                                               self.vocab_transform,  # Numericalization
                                               tensor_transform)  # Add BOS/EOS and create tensor

        return text_transform

    def get_vocab_size(self):
        SRC_VOCAB_SIZE = len(self.vocab_transform)

        return SRC_VOCAB_SIZE

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class Similarity():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained("bert-base-cased")
    def compute_score(self, real, predicted):
        encoded_real = self.tokenizer(real, return_tensors='pt')
        encoded_predicted = self.tokenizer(predicted, return_tensors='pt')
        # Make sure the input tensors are on the same device as the model (CPU or GPU)
        encoded_real = {k: v.to(self.model.device) for k, v in encoded_real.items()}
        encoded_predicted = {k: v.to(self.model.device) for k, v in encoded_predicted.items()}
        # Pass the input tensors through the model
        output_real = self.model(**encoded_real)
        output_predicted = self.model(**encoded_predicted)
        # Extract the output embeddings from the model's output
        # The output will be a tuple, and the first element contains the embeddings
        embedding_real = output_real[0]
        embedding_predicted = output_predicted[0]
        b1, w1, h1 = embedding_real.shape
        b2, w2, h2 = embedding_predicted.shape
        max_length = 200
        padding1 = torch.zeros(b1, max_length - w1, h1)
        padding2 = torch.zeros(b2, max_length - w2, h2)
        padded_real = torch.cat((embedding_real, padding1), dim=1)
        padded_predicted = torch.cat((embedding_predicted, padding2), dim=1)
        padded_real = torch.flatten(padded_real)
        padded_predicted = torch.flatten(padded_predicted)
        norm_real = torch.norm(padded_real)
        norm_predicted = torch.norm(padded_predicted)
        score = torch.dot(padded_real, padded_predicted) / (norm_real * norm_predicted)
        return score

if __name__ == '__main__':
    from torch.utils.data import DataLoader
