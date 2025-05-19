import os
import re
from collections import defaultdict
from PIL import Image

import torch
from torch import nn
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms


import torch
from torch import nn

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        captions = [item[1] for item in batch]
        captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return imgs, captions

# ------------------------------
# Vocabulary
# ------------------------------
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return text.lower().split()

    def build_vocab(self, sentence_list):
        frequencies = defaultdict(int)
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized]


# ------------------------------
# Dataset
# ------------------------------
class FlickrDataset(Dataset):
    def __init__(self, captions_dict, image_dir, transform=None, freq_threshold=5):
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids = list(captions_dict.keys())
        self.captions = []
        for k in self.image_ids:
            for cap in captions_dict[k]:
                self.captions.append((k, cap))

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab([cap for _, cap in self.captions])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        image_id, caption = self.captions[index]
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized_caption)


# ------------------------------
# Model: Encoder + Decoder
# ------------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)

    def forward(self, images):
        return self.cnn(images)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)


