import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import create_repo, list_repo_refs
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from logger import log
from src.utils.ml_utls import compute_metrics

load_dotenv()

logger = log.setup_logging()


def train(dataurl):

    # Load the datasets
    df = pd.read_csv(dataurl)

    # getting input parameter from .env file
    labels = eval(os.getenv("labels"))
    base_model_name = os.getenv("model")
    model_max_length = os.getenv("model_max_length")
    mlflow_uri = os.getenv("MLFLOW_URI")
    device = os.getenv("device")
    experiment_name = os.getenv("EXPERIMENT_NAME")
    hf_user = os.getenv("HF_USER")
    hf_token = os.getenv("HF_TOKEN")

    # deriving label2id and id2label
    label2id = dict(enumerate(labels))
    id2label = {v: k for k, v in label2id.items()}

    # Split the data into train, validation, and test sets
    train_val, test = train_test_split(df, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    # Print out the sizes of the datasets
    logger.info(
        json.dumps(
            {
                "train size": train.shape,
                "validation size": val.shape,
                "test size": test.shape,
            },
            indent=4,
        )
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Changing pandas dataframe to hugging face dataset since transformer based models doesn't work well with pandas df
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    val_dataset = Dataset.from_pandas(val)

    def tokenize(examples):
        return tokenizer(
            examples["article"], truncation=True, max_length=int(model_max_length)
        )

    train_tokenized_dataset = train_dataset.map(tokenize, batched=True).remove_columns(
        ["article", "category"]
    )
    test_tokenized_dataset = test_dataset.map(tokenize, batched=True).remove_columns(
        ["article", "category"]
    )
    val_tokenized_dataset = val_dataset.map(tokenize, batched=True).remove_columns(
        ["article", "category"]
    )

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    ).to(device)

    repo_name = f"{hf_user}/{base_model_name}-finetuned-news"

    try:
        branches = list_repo_refs(repo_name, repo_type="model", token=hf_token)
        final_branches = []
        for branch in branches.branches:
            final_branches.append(branch.name)
        if sorted(final_branches)[-1] == "main":
            model_version = "v1"
        else:
            model_version = "v" + str(int(sorted(final_branches)[-1][-1]) + 1)
    except Exception:
        create_repo(repo_name, repo_type="model", token=hf_token)
        model_version = "main"

    # Fine-tuning
    training_args = TrainingArguments(
        output_dir="./output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=1,
        logging_steps=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=val_tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model with MlFlow Tracking
    mlflow.set_registry_uri(mlflow_uri)
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.enable_system_metrics_logging()
    mlflow.transformers.autolog(log_datasets=True, log_models=True)

    try:
        # Attempt to get the experiment by name
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except AttributeError:
        # If getting the experiment raises an AttributeError, create a new one
        experiment_id = mlflow.create_experiment(experiment_name)

    mlflow.start_run(experiment_id=experiment_id)
    mlflow.log_params(
        {
            "Model Repo Name": repo_name,
            "Base Model Name": base_model_name,
            "Model Version": model_version,
        }
    )
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path=base_model_name,
    )
    trainer.train()

    model.push_to_hub(
        repo_id=repo_name,
        commit_message="Training complete!",
        revision=model_version,
        token=hf_token,
    )
    tokenizer.push_to_hub(
        repo_id=repo_name,
        commit_message="Training complete!",
        revision=model_version,
        token=hf_token,
    )

    # Evaluate the model
    results = trainer.evaluate(eval_dataset=test_tokenized_dataset)

    logger.info(f"Evaluate results: {results}")
    # Generate predictions for the test set

    predictions = trainer.predict(test_tokenized_dataset)

    test_results = test.copy(deep=True)
    test_results["label_int_pred_transfer_learning"] = np.argmax(
        predictions.predictions, axis=-1
    )
    test_results["label_pred_transfer_learning"] = test_results[
        "label_int_pred_transfer_learning"
    ].apply(lambda x: labels[x])

    accuracy = 1 - (
        len(
            test_results[
                test_results["category"] != test_results["label_pred_transfer_learning"]
            ]
        )
        / len(test_results)
    )
    logger.info(
        f"The accuracy of the fine-tuned DistilBERT transformer model on the test set is {100*accuracy:.2f}%."
    )

    mlflow.log_metric("accuracy", accuracy)

    logger.info(f"Test Accuracy: {100*accuracy:.2f}%")

    mlflow.end_run()


if __name__ == "__main__":
    try:
        processeddata_url = sys.argv[1]
        logger.info(
            f"Executing command:  python {__file__.split('/')[-1]} {processeddata_url}"
        )
        train(processeddata_url)

    except Exception as e:
        logger.error(str(e))
