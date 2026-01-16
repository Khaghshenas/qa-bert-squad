import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

def test_model_forward_pass():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    context = "Python is a programming language."
    question = "What is Python?"
    
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    
    # Check outputs contain start and end logits
    assert "start_logits" in outputs.keys() or hasattr(outputs, "start_logits")
    assert "end_logits" in outputs.keys() or hasattr(outputs, "end_logits")
    
    # Check logits shape matches input length
    assert outputs.start_logits.shape == (1, inputs["input_ids"].shape[1])
    assert outputs.end_logits.shape == (1, inputs["input_ids"].shape[1])
