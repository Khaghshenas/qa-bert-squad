import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from preprocess import preprocess

def test_preprocess_smoke():
    
    # Sample context and question  
    examples = {
        "context": ["The Eiffel Tower is located in Paris."],
        "question": ["Where is the Eiffel Tower located?"],
        "answers": [{"text": ["Paris"], "answer_start": [31]}]
    }
    
    processed = preprocess(examples)

    # Required fields
    assert "input_ids" in processed
    assert "attention_mask" in processed
    assert "start_positions" in processed
    assert "end_positions" in processed

    # Consistency checks
    assert len(processed["start_positions"]) == len(processed["end_positions"])
    assert len(processed["input_ids"]) == len(processed["start_positions"])

    # Span indices must be integers
    assert all(isinstance(i, int) for i in processed["start_positions"])
    assert all(isinstance(i, int) for i in processed["end_positions"])

    # Start/end should be valid token indices
    assert all(s >= 0 for s in processed["start_positions"])
    assert all(e >= 0 for e in processed["end_positions"])