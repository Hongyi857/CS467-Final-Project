import argparse
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from bert_context import InContextLearner

from config import *
from data import prepare_data_full
from baseline import baseline_naive_bayes
from dataset import NewsDataset
from bert_train import (
    LoRADistilBertClassifier,
    train_bert_model,
    evaluate_bert_model,
)


def run_baseline(train, test):
    clf, vec, metrics = baseline_naive_bayes(train, test)
    print("NB Accuracy:", metrics["accuracy"])
    print("NB F1 report", metrics["classification_report"])
    return


def run_bert(train, dev, test):

    model = LoRADistilBertClassifier(BERT_MODEL, num_labels=4, dropout=0.1)
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)

    # Datasets & loaders
    train_ds = NewsDataset(train, tokenizer, BERT_MAX_LEN)
    dev_ds = NewsDataset(dev, tokenizer, BERT_MAX_LEN)
    test_ds = NewsDataset(test, tokenizer, BERT_MAX_LEN)
    train_ld = DataLoader(train_ds, batch_size=BERT_BATCH, shuffle=True)
    dev_ld = DataLoader(dev_ds, batch_size=BERT_BATCH)
    test_ld = DataLoader(test_ds, batch_size=BERT_BATCH)

    train_bert_model(model, tokenizer,train_ld, dev_ld, BERT_EPOCHS, BERT_LR)
    acc, report = evaluate_bert_model(model, tokenizer,test_ld)
    print("[BERT] Test Acc:", acc)
    print("Report",report)
    return


def run_icl(train, dev):
    learner = InContextLearner(model_name=BERT_MODEL, shots=ICL_SHOTS)

    acc, report = learner.evaluate(train, dev)
    print(f"{ICL_SHOTS}-shot ICL accuracy: {acc:.2%}")
    print("Report",report)

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["baseline", "bert", "icl"], required=True)
    args = p.parse_args()

    train, dev, test = prepare_data_full()
    if args.method == "baseline":
        run_baseline(train, dev)
    elif args.method == "bert":
        run_bert(train, dev, test)
    else:
        run_icl(train, test)
