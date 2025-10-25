from torch.utils.data import Dataset
import random
import torch
import math
from loguru import logger


class AdditionDataset(Dataset):
    def __init__(self, num_data=1000, max_int=1000):
        self.num_data = num_data
        self.max_int = max_int
        self.data = self._generate_data()
        self.max_length = 3 * (math.ceil(math.log(self.max_int, 10)) + 1) + 4
        self._generate_vocab()

    def text_to_tensor(self, text):
        logger.debug(text)
        c = ""
        i = 0
        ints = []
        while i < len(text):
            c += text[i]
            if c in self.vocab:
                ints.append(self.vocab[c])
                i += 1
                c = ""
            else:
                i += 1
        return torch.tensor(ints, dtype=torch.long)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        src_text = data_dict["src"]
        tgt_text = data_dict["tgt"]

        src = self.text_to_tensor(src_text)
        tgt = self.text_to_tensor(tgt_text)
        src_length, tgt_length = src.shape[0], tgt.shape[0]
        logger.debug(f"{src_length=}, {tgt_length=}")
        seq_length = src_length + tgt_length
        pad_length = self.max_length - seq_length
        pad = torch.ones([pad_length], dtype=src.dtype) * self.pad_idx

        seq = torch.cat([src, tgt, pad], axis=0)
        logger.debug(torch.ones([src_length - 1]) * self.pad_idx)
        logger.debug(torch.ones([tgt_length]))
        logger.debug(torch.ones([pad_length + 1] * self.pad_idx))
        # mask = torch.cat([torch.ones([seq_length]), torch.zeros([pad_length])], axis=0)
        masked_tgt = torch.cat([
                                    torch.ones([src_length - 1], dtype=src.dtype) * self.pad_idx,
                                    tgt,
                                    torch.ones([pad_length + 1], dtype=src.dtype) * self.pad_idx
                                ], axis=0)

        return seq, masked_tgt

    def __len__(self):
        return self.num_data

    def vocab_size(self):
        logger.info(f"vocab_size = {len(self.vocab)}")
        return len(self.vocab)

    def _generate_vocab(self):
        self.vocab = {}
        vocab_idx = 0
        self.pad_idx = vocab_idx
        self.vocab["<pad>"] = self.pad_idx
        vocab_idx += 1
        self.sos_idx = vocab_idx
        self.vocab["<sos>"] = self.sos_idx
        vocab_idx += 1
        self.eos_idx = vocab_idx
        self.vocab["<eos>"] = self.eos_idx
        vocab_idx += 1
        for i in range(10):
            self.vocab[str(i)] = vocab_idx
            vocab_idx += 1
        self.vocab["+"] = vocab_idx
        vocab_idx += 1
        self.vocab["="] = vocab_idx
        vocab_idx += 1
        logger.debug(self.vocab)

    def _generate_data(self):
        datas = []
        for i in range(self.num_data):
            num1 = random.randrange(self.max_int)
            num2 = random.randrange(self.max_int)
            src = f"<sos>{num1}+{num2}="
            tgt = f"{num1 + num2}<eos>"
            datas.append({"src": src, "tgt": tgt})
        return datas
