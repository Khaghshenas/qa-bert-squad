#from transformers import BertTokenizerFast
from transformers import DistilBertTokenizerFast
from datasets import load_dataset

# Load tokenizer and dataset
#tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

tokenizer.save_pretrained("../models/bert-qa")

dataset = load_dataset("squad")


def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True,
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer = answers[i]

        # Character-level answer span
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        # Find context start
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1

        # Find context end
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        # If answer is not fully inside the context, label as impossible
        if not (
            offsets[context_start][0] <= start_char
            and offsets[context_end][1] >= end_char
        ):
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Find token start position
        token_start_index = context_start
        while (
            token_start_index <= context_end
            and offsets[token_start_index][0] <= start_char
        ):
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        # Find token end position
        token_end_index = context_end
        while (
            token_end_index >= context_start
            and offsets[token_end_index][1] >= end_char
        ):
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    # Remove offsets (not needed for training)
    tokenized.pop("offset_mapping")

    return tokenized


if __name__ == "__main__":
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(tokenized_dataset["train"].column_names)

    tokenized_dataset.save_to_disk("../data/squad_tokenized")
    print("Preprocessing complete.")