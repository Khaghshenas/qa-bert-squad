import torch
from datasets import load_from_disk
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import numpy as np
from collections import Counter
import string
import re

# Helper functions for EM/F1

def normalize_answer(s):

    """Lower text, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def white_space_fix(text):
        return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate():
    # Load dataset
    raw_dataset = load_from_disk("../data/squad")  # raw SQuAD dataset
    raw_val_dataset = raw_dataset["validation"].shuffle(seed=42).select(range(500))
    tokenized_dataset = load_from_disk("../data/squad_tokenized")
    val_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(500))

    # Load model and tokenizer
    model = DistilBertForQuestionAnswering.from_pretrained("../models/bert-qa")
    tokenizer = DistilBertTokenizerFast.from_pretrained("../models/bert-qa")
    #tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    model.eval()
    
    em_scores = []
    f1_scores = []


    for tokenized_example, raw_example in zip(val_dataset, raw_val_dataset):
        context = raw_example["context"]
        question = raw_example["question"]
        answers = raw_example.get("answers", {"text": [""]})
        
        # Tokenize
        #inputs = tokenizer(question, context, return_tensors="pt")
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        
        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        
        # Convert token IDs back to string
        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        answer_tokens = all_tokens[start_idx : end_idx + 1]
        prediction = tokenizer.convert_tokens_to_string(answer_tokens)
        #print(prediction)
        
        # Compute metrics
        em = max([exact_match_score(prediction, ans) for ans in answers["text"]])
        f1 = max([f1_score(prediction, ans) for ans in answers["text"]])
        
        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = np.mean(em_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    print(f"Validation Exact Match (EM): {avg_em:.2f}%")
    print(f"Validation F1 Score: {avg_f1:.2f}%")

if __name__ == "__main__":
    evaluate()
