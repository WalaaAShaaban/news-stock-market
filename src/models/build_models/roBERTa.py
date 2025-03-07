from transformers import AutoTokenizer, RobertaModel, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import pandas as pd 
import mlflow
import mlflow.pytorch

class RoBERTa:

    def __init__(self):
        print("RoBERTa Hugging face ")

    def fit(self):

        # Step 1: Load dataset
        df = pd.read_csv("..\data\processed-data\clean_news.csv")
        dataset = Dataset.from_pandas(df)
        
        # Step 2: Tokenize the dataset

        # Load RoBERTa tokenizer
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
            # Step 3: Tokenize the text data
        def tokenize_function(examples):
            return tokenizer(examples["news"], padding="max_length", truncation=True)

        # Apply tokenization to the dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # Step 4: Format the dataset for PyTorch (converts it to tensors)
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        # Step 5: Load RoBERTa model
        model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

        # Step 6: Define the training arguments (you can adjust the parameters here)
        training_args = TrainingArguments(
            output_dir="./results",  # Output directory for the model and logs
            evaluation_strategy="epoch",  # Evaluate after each epoch
            save_strategy="epoch",  # Save the model after each epoch
            per_device_train_batch_size=8,  # Batch size for training
            per_device_eval_batch_size=8,  # Batch size for evaluation
            num_train_epochs=3,  # Number of training epochs
            weight_decay=0.01,  # Weight decay for regularization
            logging_dir="./logs",  # Logging directory for TensorBoard logs
        )

        # Step 7: Set up the Trainer
        trainer = Trainer(
            model=model,  # Model to train
            args=training_args,  # Training arguments
            train_dataset=tokenized_datasets["train"],  # Training dataset
            eval_dataset=tokenized_datasets["test"],  # Evaluation dataset
        )

        # Step 8: Fine-tune the model
        trainer.train()
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Use MLflow server or local
        mlflow.set_experiment("RoBERTa Sentiment Analysis")
        # Train and log with MLflow
        with mlflow.start_run():
            mlflow.log_params(vars(training_args))  # Log training parameters
            trainer.train()
            mlflow.pytorch.log_model(model, "roberta_model")  # Save model in MLflow

        # Step 10: Save the fine-tuned model and tokenizer
        model.save_pretrained("./roberta_finetuned")
        tokenizer.save_pretrained("./roberta_finetuned")



    def predict(self):
        
        # Load the fine-tuned model from MLflow
        model_path = "roberta_model"
        loaded_model = mlflow.pytorch.load_model(model_path)
        text = "This movie was amazing! The story was fantastic and the actors did a great job."
        result = loaded_model(text)

        print(result)  # Output: [{'label': 'POSITIVE', 'score': 0.99}]






