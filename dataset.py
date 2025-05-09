import torch
from torch.utils.data import Dataset
from config import BERT_MAX_LEN


class NewsDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=BERT_MAX_LEN):
        """
        data: a Hugging Face dataset split (list/dict-like) with keys "text" and "label".
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]["text"]
        label = self.data[index]["label"]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: encoding[key].squeeze(0) for key in encoding}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
