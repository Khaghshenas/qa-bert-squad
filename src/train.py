from datasets import load_from_disk
from transformers import BertForQuestionAnswering, Trainer, TrainingArguments
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

def train():
    dataset = load_from_disk("../data/squad_tokenized")

    print("Train columns:", dataset["train"].column_names)
    print("Sample:", dataset["train"][0])

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            #"token_type_ids", #needed for full Bert
            "start_positions",
            "end_positions",
        ],
    )

    #model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    training_args = TrainingArguments(
        output_dir="../models/bert-qa",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_dir="../logs",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # 
    # For demonstration and local CPU experiments, we use a smaller subset of the dataset (5k training examples instead of all 88k).
    # This drastically reduces training time. Full dataset training is recommended on a GPU.
    small_train = dataset["train"].shuffle(seed=42).select(range(5000))
    small_valid = dataset["validation"].shuffle(seed=42).select(range(500))

    trainer = Trainer(
        model=model,
        args=training_args,
        #train_dataset=dataset["train"],
        #eval_dataset=dataset["validation"],
        train_dataset=small_train,
        eval_dataset=small_valid,
    )

    trainer.train()
    model.save_pretrained("../models/bert-qa")

if __name__ == "__main__":
    train()
