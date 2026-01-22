from transformers import DistilBertTokenizerFast
from datasets import load_dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load tokenizer and dataset
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained(os.path.join(BASE_DIR, "../models/bert-qa"))

dataset = load_dataset("squad")

def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="longest",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mappings = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mappings):
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]

        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        try:
            context_start = next(j for j, s_id in enumerate(sequence_ids) if s_id == 1)
            rev_index = next(j for j, s_id in enumerate(reversed(sequence_ids)) if s_id == 1)
            context_end = len(sequence_ids) - 1 - rev_index
        except StopIteration:
            start_positions.append(0)
            end_positions.append(0)
            continue

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start_index = context_start
        while token_start_index <= context_end and offsets[token_start_index][0] < start_char:
            token_start_index += 1

        token_end_index = context_end
        while token_end_index >= context_start and offsets[token_end_index][1] > end_char:
            token_end_index -= 1

        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized

if __name__ == "__main__":
    # Important: assign the result back
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(tokenized_dataset["train"].column_names)

    # Save to disk
    tokenized_dataset.save_to_disk(os.path.join(BASE_DIR, "../data/squad_tokenized"))
    print("Preprocessing complete and dataset saved.")