import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from preprocess import preprocess

def test_preprocess():
    # Sample context and question  
    examples = {
        "context": ["The Eiffel Tower is located in Paris."],
        "question": ["Where is the Eiffel Tower located?"],
        "answers": [{"text": ["Paris"], "answer_start": [31]}]
    }

    # Call preprocess
    tokenized = preprocess(examples)

    # Check that start_positions and end_positions exist
    assert "start_positions" in tokenized
    assert "end_positions" in tokenized

    # Check that lengths match input batch
    assert len(tokenized["start_positions"]) == len(examples["context"])
    assert len(tokenized["end_positions"]) == len(examples["context"])

    # Check input_ids and attention_mask exist
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
