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

    """tokenized = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True,
    )"""

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

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = tokenized.sequence_ids(i)

        # context start & end in this chunk
        context_start = next((j for j, s_id in enumerate(sequence_ids) if s_id == 1), None)
        context_end = len(sequence_ids) - 1 - next((j for j, s_id in enumerate(reversed(sequence_ids)) if s_id == 1), None)
        if context_start is None or context_end is None:
            start_positions.append(0)
            end_positions.append(0)
            continue

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
            continue

        # token start
        token_start_index = context_start
        while token_start_index <= context_end and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        # token end
        token_end_index = context_end
        while token_end_index >= context_start and offsets[token_end_index][1] >= end_char:
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