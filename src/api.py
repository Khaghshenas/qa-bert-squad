from flask import Flask, request, jsonify
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import torch

app = Flask(__name__)

model = DistilBertForQuestionAnswering.from_pretrained("../models/bert-qa")
tokenizer = DistilBertTokenizerFast.from_pretrained("../models/bert-qa")

model.eval()

@app.route("/predict", methods=["POST"])
def predict():
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