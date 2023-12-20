from collections import Counter
from itertools import chain
from typing import List

import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset


class Vocabularizer:
    def __init__(self, 
                 tokenized_texts: List[List[str]], 
                 max_vocab_size=None):
        """
        Builds a vocabulary by concatenating all tokenized texts and counting words.
        Most common words are placed in vocabulary, others are replaced with [UNK] token
        :param tokenized_texts: texts to build a vocab
        :param max_vocab_size: amount of words in vocabulary
        """
        counts = Counter(chain(*tokenized_texts))
        max_vocab_size = max_vocab_size or len(counts)
        common_pairs = counts.most_common(max_vocab_size)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.EOS_IDX = 2
        self.itos = ["<PAD>", "<UNK>", "<EOS>"] + [pair[0] for pair in common_pairs]
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def vectorize(self, text: List[str]):
        """
        Maps each token to it's index in the vocabulary
        :param text: sequence of tokens
        :return: vectorized sequence
        """
        return [self.stoi.get(tok, self.UNK_IDX) for tok in text]

    def __iter__(self):
        return iter(self.itos)

    def __len__(self):
        return len(self.itos)


def w2v_embeds(model, vocab: Vocabularizer):
    mean = model.vectors.mean(1).mean()
    std = model.vectors.std(1).mean()
    vec_size = model.vector_size
    emb_matrix = torch.zeros((len(vocab), vec_size))
    for i, word in enumerate(vocab.itos[1:], 1):
        try:
            for j, name in zip(model.key_to_index.values(), model.key_to_index.keys()):
                if name.find(word) != -1:
                    emb_matrix[i] = torch.tensor(model[j])
                    break
        except KeyError:
            emb_matrix[i] = torch.randn(vec_size) * std + mean
    return emb_matrix

def tfidf_embeds(TfIdf_model, data, vocabulary: Vocabularizer):
    res = TfIdf_model.transform(data).toarray().T
    vec_size = res.shape[1]
    embeddings = torch.zeros((len(vocabulary), vec_size))
    for i, word in enumerate(vocabulary.itos[1:], 1):
        try:
            id = TfIdf_model.get_feature_names_out().tolist().index(word)
            embeddings[i] = torch.tensor(TfIdf_model.transform(id))
        except:
            embeddings[i] = torch.randn(vec_size)
    return embeddings


class DatasetLoader(Dataset):
    def __init__(self, 
                 tokenized_texts, 
                 labels, 
                 vocab: Vocabularizer):
        """
        A Dataset for the task
        :param tokenized_texts: texts from a train/val/test split
        :param labels: corresponding ratings
        :param vocab: vocabulary with indexed tokens
        """
        self.texts = tokenized_texts
        self.labels = labels
        self.vocab = vocab

    def __getitem__(self, item):
        return (
            self.vocab.vectorize(self.texts[item]) + [self.vocab.EOS_IDX],
            self.labels[item],
        )

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        """
        Technical method to form a batch to feed into recurrent network
        """
        tmp = pack_sequence([torch.tensor(pair[0]) for pair in batch], enforce_sorted=False
        ), torch.tensor([pair[1] for pair in batch])
        return tmp


