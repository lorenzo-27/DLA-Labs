import argparse
import torch
from datasets import load_dataset, get_dataset_split_names
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy as np
import wandb
import evaluate
from rich.console import Console

console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BERT fine-tuning for sentiment analysis')

    # General parameters
    parser.add_argument('--project', type=str, default='BERT-Sentiment-Analysis',
                        help='WandB project name')
    parser.add_argument('--dataset', type=str, default='cornell-movie-review-data/rotten_tomatoes',
                        help='HuggingFace dataset to use')
    parser.add_argument('--model', type=str, default='distilbert/distilbert-base-uncased',
                        help='Pre-trained model to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cpu/cuda)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--exercise', type=int, default=1, choices=[1, 2, 3],
                        help='Exercise to run (1=baseline, 2=finetuning, 3=efficient finetuning)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for tokenization')

    # Exercise 3 parameters for efficient fine-tuning
    parser.add_argument('--method', type=str, default='lora',
                        choices=['lora', 'freeze', 'adapters'],
                        help='Method for efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA r dimension')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout rate')

    return parser.parse_args()


def tokenize_dataset(tokenizer, dataset, max_length):
    """Tokenize dataset using the provided tokenizer."""

    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

    tokenized_dataset = {}
    for split in dataset:
        tokenized_dataset[split] = dataset[split].map(
            preprocess_function,
            batched=True
        )

    return tokenized_dataset


def compute_metrics(eval_pred):
    """Compute evaluation metrics for the trainer."""
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)['accuracy']
    f1 = load_f1.compute(predictions=predictions, references=labels, average='weighted')['f1']

    return {"accuracy": accuracy, "f1": f1}


def exercise1_baseline(args, dataset, model, tokenizer):
    """Run Exercise 1: SVM baseline with BERT features."""
    console.print("[bold blue]Running Exercise 1: SVM baseline with BERT features[/bold blue]")

    # Initialize the feature extraction pipeline
    extractor = AutoModel.from_pretrained(args.model)
    extractor.to(args.device)

    # Extract features from datasets
    def extract_features(dataset_split):
        features = []
        labels = []

        for batch_idx in range(0, len(dataset_split), args.batch_size):
            batch = dataset_split[batch_idx:batch_idx + args.batch_size]
            inputs = tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt'
            ).to(args.device)

            with torch.no_grad():
                outputs = extractor(**inputs)
                # Extract [CLS] token from the last hidden state
                cls_features = outputs.last_hidden_state[:, 0].cpu().numpy()

            features.append(cls_features)
            labels.extend(batch['label'])

        return np.vstack(features), np.array(labels)

    console.print("[blue]Extracting features from training set...[/blue]")
    X_train, y_train = extract_features(dataset['train'])

    console.print("[blue]Extracting features from validation set...[/blue]")
    X_val, y_val = extract_features(dataset['validation'])

    console.print("[blue]Extracting features from test set...[/blue]")
    X_test, y_test = extract_features(dataset['test'])

    # Train a linear SVM
    console.print("[blue]Training SVM classifier...[/blue]")
    svm = LinearSVC(random_state=args.seed)
    svm.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = svm.predict(X_val)
    val_report = classification_report(y_val, val_preds, output_dict=True)

    # Evaluate on test set
    test_preds = svm.predict(X_test)
    test_report = classification_report(y_test, test_preds, output_dict=True)

    console.print("[bold green]Validation set metrics:[/bold green]")
    console.print(f"Accuracy: {val_report['accuracy']:.4f}")
    console.print(f"F1 Score: {val_report['weighted avg']['f1-score']:.4f}")

    console.print("[bold green]Test set metrics:[/bold green]")
    console.print(f"Accuracy: {test_report['accuracy']:.4f}")
    console.print(f"F1 Score: {test_report['weighted avg']['f1-score']:.4f}")

    # Log to wandb
    wandb.log({
        "val_accuracy": val_report['accuracy'],
        "val_f1": val_report['weighted avg']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_f1": test_report['weighted avg']['f1-score']
    })

    return {
        "val_accuracy": val_report['accuracy'],
        "val_f1": val_report['weighted avg']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_f1": test_report['weighted avg']['f1-score']
    }


def exercise2_finetuning(args, tokenized_dataset, tokenizer):
    """Run Exercise 2: Full fine-tuning of BERT for sentiment analysis."""
    console.print("[bold blue]Running Exercise 2: Full fine-tuning of BERT[/bold blue]")

    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2
    )

    # Data collator for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{args.project}/full-finetuning",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb" if wandb.run is not None else None,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    console.print("[blue]Starting fine-tuning...[/blue]")
    trainer.train()

    # Evaluate on test set
    console.print("[blue]Evaluating on test set...[/blue]")
    test_results = trainer.evaluate(tokenized_dataset['test'])

    console.print("[bold green]Test set metrics:[/bold green]")
    console.print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    console.print(f"Test F1 Score: {test_results['eval_f1']:.4f}")

    return test_results


def exercise3_efficient_finetuning(args, tokenized_dataset, tokenizer):
    """Run Exercise 3: Efficient fine-tuning of BERT for sentiment analysis."""
    console.print(f"[bold blue]Running Exercise 3: Efficient fine-tuning with {args.method}[/bold blue]")

    if args.method == 'lora':
        # Use LoRA (Low-Rank Adaptation) for efficient fine-tuning
        from peft import LoraConfig, get_peft_model, TaskType

        # Load base model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2
        )

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
            bias="none",
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif args.method == 'freeze':
        # Freeze most of the model, only train classifier layers
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2
        )

        # Freeze all layers except classifier
        for param in model.base_model.parameters():
            param.requires_grad = False

        # Print trainable parameters percentage
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"Trainable parameters: {trainable_params} ({trainable_params / total_params:.2%})")

    elif args.method == 'adapters':
        # Use adapter modules
        from adapters import AutoAdapterModel

        # Load model with adapter support
        model = AutoAdapterModel.from_pretrained(args.model)

        # Add a classification head
        model.add_classification_head("sentiment", num_labels=2)

        # Add a new adapter
        model.add_adapter("sentiment_adapter")

        # Activate the adapter
        model.train_adapter("sentiment_adapter")

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        console.print(f"Trainable parameters: {trainable_params} ({trainable_params / total_params:.2%})")

    # Data collator for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{args.project}/{args.method}",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch", # evaluation_strategy is deprecated amd will be removed in transformers 4.46
        save_strategy="epoch",
        load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
        # greater_is_better=True,
        report_to="wandb" if wandb.run is not None else None,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        processing_class=tokenizer, # tokenizer is deprecated and will be removed in transformers 5.0.0
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    console.print("[blue]Starting efficient fine-tuning...[/blue]")
    trainer.train()

    # Evaluate on test set
    console.print("[blue]Evaluating on test set...[/blue]")
    test_results = trainer.evaluate(tokenized_dataset['test'])

    console.print("[bold green]Test set metrics:[/bold green]")
    console.print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    console.print(f"Test F1 Score: {test_results['eval_f1']:.4f}")

    return test_results


def main():
    """Main function."""
    args = parse_args()
    console = Console()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize wandb
    wandb_run = wandb.init(
        project=args.project,
        config={
            'dataset': args.dataset,
            'model': args.model,
            'exercise': args.exercise,
            'method': args.method if args.exercise == 3 else None,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'max_length': args.max_length,
            'lora_r': args.lora_r if args.method == 'lora' else None,
            'lora_alpha': args.lora_alpha if args.method == 'lora' else None,
            'lora_dropout': args.lora_dropout if args.method == 'lora' else None,
            'seed': args.seed
        }
    )

    # Load dataset
    console.print(f"[blue]Loading dataset {args.dataset}...[/blue]")
    splits = get_dataset_split_names(args.dataset)
    console.print(f"Dataset splits: {splits}")

    dataset = load_dataset(args.dataset)
    console.print(f"Dataset loaded: {dataset}")

    # Load tokenizer
    console.print(f"[blue]Loading tokenizer for {args.model}...[/blue]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Run the selected exercise
    if args.exercise == 1:
        # Exercise 1: SVM baseline
        model = AutoModel.from_pretrained(args.model)
        results = exercise1_baseline(args, dataset, model, tokenizer)
    else:
        # Tokenize dataset for exercises 2 and 3
        console.print("[blue]Tokenizing dataset...[/blue]")
        tokenized_dataset = tokenize_dataset(tokenizer, dataset, args.max_length)

        if args.exercise == 2:
            # Exercise 2: Full fine-tuning
            results = exercise2_finetuning(args, tokenized_dataset, tokenizer)
        elif args.exercise == 3:
            # Exercise 3: Efficient fine-tuning
            results = exercise3_efficient_finetuning(args, tokenized_dataset, tokenizer)

    # Close wandb run
    wandb_run.finish()

    return results


if __name__ == "__main__":
    main()