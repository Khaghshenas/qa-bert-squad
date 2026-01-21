import os
from pathlib import Path
import pytest
from datasets import load_from_disk
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "squad_tokenized"

def test_training_pipeline_smoke():
    """
    This test ensures the training pipeline initializes correctly without running model training.
    """
    if os.getenv("CI") and not DATASET_PATH.exists():
        pytest.skip("Dataset not available in CI environment")

    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"

    # Load dataset (just check it loads)
    dataset = load_from_disk(DATASET_PATH)
    assert "train" in dataset
    assert "validation" in dataset

    # Format dataset
    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
        ],
    )

    # Use a very small subset
    train_subset = dataset["train"].select(range(2))
    valid_subset = dataset["validation"].select(range(2))

    # Load model and tokenizer from base, as our trained model does not exist in CI
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments (no real training)
    training_args = TrainingArguments(
        output_dir="/tmp/test-output",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
    )


    # Initialize Trainer (key check!)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=valid_subset,
        data_collator=data_collator,
    )

    # Do NOT call trainer.train()
    assert trainer is not None
