import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, AdamW
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
from torch.nn.utils import clip_grad_norm_
from transformers import DistilBertTokenizer, DistilBertModel
from peft import LoraConfig, get_peft_model

class LoRADistilBertClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=4, dropout=0.1):
        super().__init__()
        # load base DistilBERT
        self.bert = DistilBertModel.from_pretrained(model_name)
        # apply LoRA
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_lin", "k_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none",
        )
        # apply PEFT model
        self.bert = get_peft_model(self.bert, lora_cfg)
        # classification head
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Retrieve the token that represents the sequence
        hidden_states = outputs.last_hidden_state  
        pooled_output = hidden_states[:, 0]  
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_bert_model(
    model, tokenizer,train_loader, dev_loader,num_epochs=3, learning_rate=2e-5
):
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # Wrap the train_loader with tqdm for a progress bar
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        )
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            # Update progress bar with current loss value
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        print(f"[BERT] Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluate on the development set after each epoch
        dev_accuracy, report = evaluate_bert_model(model, tokenizer, dev_loader)
        print(f"[BERT] Development Set Accuracy: {dev_accuracy:.4f}")
        print("Classification Report:\n", report)


def evaluate_bert_model(model, tokenizer,data_loader):
    model.eval()
    errors = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            # For error analysis
            for i in range(len(labels)):
                if predictions[i] != labels[i]:

                    decoded_text = tokenizer.decode(
                        input_ids[i], skip_special_tokens=True
                    )
                    errors.append(
                        (decoded_text, labels[i].item(), predictions[i].item())
                    )
    if len(errors) >= 3:
        sample_errors = random.sample(errors, 3)
        print("\nSample Misclassified Examples:")
        for text, true_label, pred_label in sample_errors:
            print(
                f"Text: {text}\nTrue Label: {true_label}, Predicted Label: {pred_label}\n"
            )
    elif len(errors) > 0:
        print("\nMisclassified Examples:")
        for text, true_label, pred_label in errors:
            print(
                f"Text: {text}\nTrue Label: {true_label}, Predicted Label: {pred_label}\n"
            )
    else:
        print("\nNo misclassified examples found.")
    model.train()  # Go back to training
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["World","Sports","Business","Sci/Tech"],
        digits=4,
        zero_division=0,
    )
    return acc, report


