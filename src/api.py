from pathlib import Path
from flask import Flask, request, jsonify
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import torch
import os
from unittest.mock import MagicMock

app = Flask(__name__)

# Configurable model path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bert-qa/"
MODEL_PATH = str(MODEL_PATH)
print("Model path resolved to:", MODEL_PATH)
IS_CI = os.getenv("CI") == "true"

# Globals for lazy loading
model, tokenizer = None, None

def get_model_and_tokenizer():
    """Load the real model or return a dummy model in CI."""
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return model, tokenizer

    if IS_CI:
        # Dummy model for CI/testing
        class DummyOutput:
            def __init__(self):
                self.start_logits = torch.tensor([[0, 0, 0, 0, 10]])
                self.end_logits = torch.tensor([[0, 0, 0, 0, 10]])

        class DummyModel:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                return DummyOutput()

        class DummyTokenizer:
            def __call__(self, question, context, return_tensors="pt", truncation=True, max_length=384):
                # Return a minimal fake input similar to what HuggingFace tokenizer returns
                return {
                    "input_ids": torch.tensor([[101, 102, 103, 104, 105]]),  # dummy token IDs
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
                }

            def decode(self, ids, skip_special_tokens=True):
                return "Paris"

        model, tokenizer = DummyModel(), DummyTokenizer()
        return model, tokenizer

    # Load real model
    model_dir = Path(MODEL_PATH)
    valid_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "flax_model.msgpack"]
    if not model_dir.exists() or not any((model_dir / f).exists() for f in valid_files):
       raise FileNotFoundError(f"Model not found at {model_dir.resolve()}")

    model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model.eval()

    return model, tokenizer


@app.route("/predict", methods=["POST"])
def predict():
    model, tokenizer = get_model_and_tokenizer()

    data = request.json
    question = data.get("question")
    context = data.get("context")

    if question is None or context is None:
        return jsonify({"error": "Provide 'question' and 'context'"}), 400

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits, dim=-1).item()
    end_idx = torch.argmax(outputs.end_logits, dim=-1).item()

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx : end_idx + 1],
        skip_special_tokens=True,
    )

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)