import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments, 
)

type AnyTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

def preprocess(examples: dict,  tokenizer: AnyTokenizer) -> BatchEncoding:
    """
    Tokenizes input text and returns a BatchEncoding object.
    """
    return tokenizer(examples["text"], truncation=True, padding=True)

def save_model(model: AutoModelForSequenceClassification, 
               tokenizer: AutoTokenizer,
               filename: str) -> None:
    model.save_pretrained(filename)
    tokenizer.save_pretrained(filename)
    print(f"Training complete. Model saved to {filename}")

def setup() -> tuple[str, AutoModelForSequenceClassification]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification\
        .from_pretrained(model_name, num_labels=2)\
        .to(device)
    return (model_name, model)

def train_model() -> None:
    model_name, model = setup()

    # Load the Dataset (Deepset's Prompt Injection dataset)
    # This contains 'label' 1 for injection and 0 for benign
    dataset = load_dataset("deepset/prompt-injections")

    # Preprocess
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_data = dataset.map(
         preprocess,
         batched=True,
         fn_kwargs={"tokenizer": tokenizer}
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training Arguments optimized for 6GB VRAM
    training_args = TrainingArguments(
        output_dir="./sentinel_model_v1",
        learning_rate=2e-5,
        per_device_train_batch_size=8, # Adjust to 8 if you get "Out of Memory"
        per_device_eval_batch_size=8,  # Add this to keep eval memory low too
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True, # NOTE: Uses Tensor Cores on my RTX 3060 for 2x speed
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training on GPU...")
    trainer.train()
    
    save_model(model, tokenizer, "./fine_tuned_sentinel")

if __name__ == "__main__":
    train_model()