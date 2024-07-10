import os
import json
from typing import Dict, List, Tuple
import numpy as np
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sentence_transformers.losses import CosineSimilarityLoss

# Configuration
DATASETS = {
    "sst5": ("SetFit/sst5", "label", ["very negative", "negative", "neutral", "positive", "very positive"]),
    "cr": ("SetFit/SentEval-CR", "label", ["negative", "positive"]),
    "emotion": ("SetFit/emotion", "label", ["sadness", "joy", "love", "anger", "fear", "surprise"]),
    "enron_spam": ("SetFit/enron_spam", "label", ["not_spam", "spam"]),
    "ag_news": ("SetFit/ag_news", "label", ["World", "Sports", "Business", "Sci/Tech"])
}

FEW_SHOT_SIZES = [8, 64]
MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"  # As used in the paper
NUM_SPLITS = 10  # Number of random splits as per the paper

def load_config(config_path: str) -> Dict:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results: Dict, output_path: str):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def prepare_dataset(dataset_name: str, num_samples: int, seed: int) -> Tuple[Dict, Dict, List]:
    """Prepare dataset for training and testing with a specific random seed."""
    dataset_info = DATASETS[dataset_name]
    dataset = load_dataset(dataset_info[0])
    train_dataset = sample_dataset(dataset["train"], label_column=dataset_info[1], num_samples=num_samples, seed=seed)
    test_dataset = dataset["test"]
    return train_dataset, test_dataset, dataset_info[2]

def train_and_evaluate(dataset_name: str, num_samples: int, config: Dict, seed: int) -> Dict:
    """Train and evaluate the model for a specific dataset, few-shot size, and random seed."""
    train_dataset, test_dataset, labels = prepare_dataset(dataset_name, num_samples, seed)
    
    model = SetFitModel.from_pretrained(MODEL_NAME, labels=labels)
    
    args = TrainingArguments(
        batch_size=config.get("batch_size", 16),  # As per paper
        num_epochs=config.get("num_epochs", 1),  # As per paper
        head_learning_rate=config.get("learning_rate", 1e-3),  # As per paper
        max_length=256,  # As per paper
        loss=CosineSimilarityLoss  # As per paper
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    # Evaluation
    predictions = model.predict(test_dataset["text"])
    accuracy = accuracy_score(test_dataset["label"], predictions)
    
    # For binary classification, also compute Matthews Correlation Coefficient
    if len(labels) == 2:
        mcc = matthews_corrcoef(test_dataset["label"], predictions)
        return {"accuracy": accuracy, "matthews_correlation": mcc}
    else:
        return {"accuracy": accuracy}

def run_experiment(config_path: str, output_path: str):
    """Run the complete experiment for all datasets and few-shot sizes."""
    config = load_config(config_path)
    results = {}
    
    for dataset_name in DATASETS.keys():
        results[dataset_name] = {}
        for num_samples in FEW_SHOT_SIZES:
            print(f"Running experiment for {dataset_name} with {num_samples} samples per class...")
            metrics = []
            for seed in range(NUM_SPLITS):
                iteration_metrics = train_and_evaluate(dataset_name, num_samples, config, seed)
                metrics.append(iteration_metrics)
            
            # Calculate average and standard deviation
            avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
            std_metrics = {k: np.std([m[k] for m in metrics]) for k in metrics[0]}
            
            results[dataset_name][f"{num_samples}_shot"] = {
                "avg": avg_metrics,
                "std": std_metrics
            }
    
    save_results(results, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    config_path = "setfit_config.json"
    output_path = "setfit_results.json"
    run_experiment(config_path, output_path)