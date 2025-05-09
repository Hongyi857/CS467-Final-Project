from datasets import load_dataset
from collections import Counter
import pandas as pd
import re
from config import HF_TOKEN, DEV_SIZE, SEED


def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def preprocess_dataset(dataset):
    df = pd.DataFrame(dataset)
    df["text"] = df["text"].apply(clean_text)
    return df


def prepare_data_full(dev_size: int = DEV_SIZE, seed: int = SEED):
    dataset = load_dataset("fancyzhx/ag_news", token=HF_TOKEN)
    train_orig, test_orig = dataset["train"], dataset["test"]
    split = train_orig.train_test_split(
        test_size=dev_size, stratify_by_column="label", seed=seed
    )
    new_train, new_dev = split["train"], split["test"]
    # shuffle
    new_train = new_train.shuffle(seed=seed)
    new_dev = new_dev.shuffle(seed=seed)
    test_orig = test_orig.shuffle(seed=seed)

    # new_train = new_train.select(range(3000))
    # new_dev = new_dev.select(range(240))
    # test_orig = test_orig.select(range(3000))

    print("Sizes:", len(new_train), len(new_dev), len(test_orig))
    print("Train dist:", Counter(new_train["label"]))
    print("Dev   dist:", Counter(new_dev["label"]))
    print("Test  dist:", Counter(test_orig["label"]))

    return new_train, new_dev, test_orig
