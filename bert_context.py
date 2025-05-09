import random
import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from config import BERT_MODEL, ICL_SHOTS
from tqdm import tqdm
from sklearn.metrics import classification_report


class InContextLearner:
    """
    Encapsulates in-context learning logic using a frozen DistilBERT MLM.
    """

    def __init__(
        self, model_name: str = BERT_MODEL, shots: int = ICL_SHOTS, seed: int = 42
    ):
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForMaskedLM.from_pretrained(model_name)
        self.model.eval()

        # Few-shot parameters
        self.shots = shots
        random.seed(seed)

        # Label mappings
        self.label_str = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Science",
        }
        # Map label IDs to single-token IDs
        self.label_tokens = {
            label: self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(lbl_str)[0]
            )
            for label, lbl_str in self.label_str.items()
        }

    def build_prompt(self, demos: list[dict], query_text: str) -> str:
        """
        Construct a cloze-style prompt from demonstration examples and a query text.
        """
        prompt = ""
        for d in demos:
            prompt += f"News article: {d['text']}\n Genre: {d['label']}\n"
        prompt += f"News article: {query_text}\n Genre: {self.tokenizer.mask_token}\n"
        return prompt

    def predict_label(self, prompt: str) -> int:
        """
        Run the model on the prompt, return the label ID with highest masked-token score.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        mask_bool = (inputs["input_ids"][0] == self.tokenizer.mask_token_id)

        mask_positions = mask_bool.nonzero(as_tuple=True)[0].item()

        with torch.no_grad():
            logits = self.model(**inputs).logits[0, mask_positions]

        scores = {
            label: logits[token_id].item() for label, token_id in self.label_tokens.items()
        }
        return max(scores, key=scores.get)

    def evaluate(self, train_data, eval_data):
        """
        Perform k‑shot in‑context evaluation *without* reusing any support example
        across the entire evaluation set. Returns (accuracy, classification_report).
        """
        train_list = list(train_data)
        total      = len(eval_data)
        needed     = self.shots * total

        if needed > len(train_list):
            raise ValueError(
                f"Not enough training examples ({len(train_list)}) "
                f"to sample {self.shots} × {total} = {needed} without replacement."
            )

        sampled = random.sample(train_list, needed)

        correct = 0
        y_true = []
        y_pred = []

        for i, example in enumerate(tqdm(eval_data, desc="Evaluating", leave=False)):
            if self.shots > 0:
                start = i * self.shots
                end   = start + self.shots
                demos = sampled[start:end]

                demos = [
                    {"text": d["text"], "label": self.label_str[d["label"]]}
                    for d in demos
                ]

                prompt = self.build_prompt(demos, example["text"])
            else:
                prompt = f"Text: {example['text']} Label: {self.tokenizer.mask_token}."

            pred = self.predict_label(prompt)
            y_pred.append(pred)
            y_true.append(example["label"])

            if pred == example["label"]:
                correct += 1

        accuracy = correct / total
        report = classification_report(
            y_true,
            y_pred,
            target_names=["World", "Sports", "Business", "Sci/Tech"],
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        return accuracy, report
