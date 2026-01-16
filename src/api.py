import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from transformers import DistilBertForQuestionAnswering, BertForQuestionAnswering, DistilBertTokenizerFast, BertTokenizer
import torch

app = Flask(__name__)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("models/bert-qa")

        
#@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    inputs = tokenizer(data["question"], data["context"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1])
    )
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000)

