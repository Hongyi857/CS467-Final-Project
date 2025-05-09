# CS467-Final-Project
Dataset: https://huggingface.co/datasets/fancyzhx/ag_news
# Run Naive Bayes baseline
python main.py --method baseline

# Train and evaluate DistilBERT with LoRA fine-tuning
python main.py --method bert

# Run DistilBERT in-context learning (1-shot setup)
python main.py --method icl
