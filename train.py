import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os


class MedicalNoteDataset(Dataset):
    """Dataset class for medical notes with labels"""
    def __init__(self, texts, labels, tokenizer, max_length=4096):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Convert label to text: 1 -> "Epilepsy", 0 -> "PNES"
        label_text = "Epilepsy" if label == 1 else "PNES"
        
        # Create instruction-following format
        prompt = f"""Based on the patient's clinical note, classify if the patient has Epilepsy or PNES (Psychogenic Non-Epileptic Seizures).

If the prediction is Epilepsy, output 'Epilepsy'. If the prediction is PNES, output 'PNES'.
Output either 'Epilepsy' or 'PNES', nothing else.

Patient's clinical note: {text}

Answer:"""
        
        # Format as chat messages
        messages = [
            {"role": "system", "content": "You are a neurologist. Follow instructions exactly."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label_text}
        ]
        
        # Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()  # For causal LM, labels are same as input_ids
        }


class EpilepyPNESClassifier:
    def __init__(self, model_name="meta-llama/Llama-3.3-70B-Instruct", 
                 mode="train", 
                 learning_rate=2e-4,
                 num_epochs=3,
                 batch_size=1,
                 gradient_accumulation_steps=8,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.05,
                 output_dir="./llama_qlora_checkpoints",
                 local_model_path=None):
        """
        Initialize the classifier
        
        Args:
            model_name: HuggingFace model name (used as fallback)
            mode: "train" or "test"
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            gradient_accumulation_steps: Steps to accumulate gradients
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            local_model_path: Path to local model directory (NEW)
        """
        self.model_name = model_name
        self.mode = mode
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.output_dir = output_dir
        self.local_model_path = local_model_path
        
        # Use local path if provided, otherwise use model_name
        model_path = self.local_model_path if self.local_model_path else self.model_name
        
        print(f"Mode: {mode}")
        print(f"Loading model from: {model_path}")
        
        # Configure 4-bit quantization for QLoRA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer from local path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True if self.local_model_path else False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model with quantization from local path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True if self.local_model_path else False
        )
        
        if mode == "train":
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        elif mode == "test":
            # Load fine-tuned weights if available
            adapter_path = os.path.join(self.output_dir, "final_model")
            if os.path.exists(adapter_path):
                print(f"Loading fine-tuned adapter from {adapter_path}")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            else:
                print(f"Warning: No fine-tuned model found at {adapter_path}")
                print("Using base model for inference")
        
        print("Model loaded successfully!")
    
    def train(self, train_dataset, val_dataset):
        """
        Train the model using QLoRA
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",  # Use eval_strategy
            do_eval=True,  # Explicitly enable evaluation
            fp16=True,
            optim="paged_adamw_8bit",  # Memory-efficient optimizer
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        print("\nStarting training...")
        train_result = trainer.train()
        
        # Save the final model
        final_model_path = os.path.join(self.output_dir, "final_model")
        trainer.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        print(f"\nModel saved to {final_model_path}")
        
        # Extract and plot training loss
        self.plot_training_loss(trainer.state.log_history)
        
        return train_result
    
    def plot_training_loss(self, log_history):
        """
        Plot and save training loss curve
        
        Args:
            log_history: Training log history from trainer
        """
        train_losses = []
        eval_losses = []
        steps = []
        eval_steps = []
        
        for log in log_history:
            if 'loss' in log:
                train_losses.append(log['loss'])
                steps.append(log['step'])
            if 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])
                eval_steps.append(log['step'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_losses, label='Training Loss', marker='o')
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='s')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_plot_path = os.path.join(self.output_dir, 'training_loss_curve.png')
        plt.savefig(loss_plot_path, dpi=300)
        print(f"Training loss curve saved to {loss_plot_path}")
        plt.close()
    
    def generate_prompt(self, patient_note: str) -> str:
        """Generate prompt for inference"""
        return f"""Based on the patient's clinical note, classify if the patient has Epilepsy or PNES (Psychogenic Non-Epileptic Seizures).

If the prediction is Epilepsy, output 'Epilepsy'. If the prediction is PNES, output 'PNES'.
Output either 'Epilepsy' or 'PNES', nothing else.

Patient's clinical note: {patient_note}

Answer:"""
    
    def predict(self, patient_note: str):
        """
        Predict label for a single patient note
        
        Args:
            patient_note: Clinical note text
            
        Returns:
            Prediction (1 for Epilepsy, 0 for PNES, None if invalid)
        """
        try:
            prompt = self.generate_prompt(patient_note)
            messages = [
                {"role": "system", "content": "You are a neurologist. Follow instructions exactly."},
                {"role": "user", "content": prompt}
            ]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=10,
                    temperature=0,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            input_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            if input_text in response:
                response = response.replace(input_text, "").strip()
            
            # Parse response
            if 'Epilepsy' in response:
                return 1
            if 'PNES' in response:
                return 0
            
            return None  # Skip if neither found
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def process_excel_file(self, input_file: str, output_file: str, 
                          note_column: str = 'NOTE_TEXT', 
                          label_column: str = 'Label',
                          split_column: str = 'Split',
                          train_ratio: float = 1.0):
        """
        Process Excel file for training or testing using pre-defined splits
        
        Args:
            input_file: Path to input Excel file
            output_file: Path to output Excel file
            note_column: Name of column containing clinical notes
            label_column: Name of column containing labels
            split_column: Name of column containing split information
        """
        print(f"Reading Excel file: {input_file}")
        df = pd.read_excel(input_file)
        
        if note_column not in df.columns:
            raise ValueError(f"Column '{note_column}' not found")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found")
        if split_column not in df.columns:
            raise ValueError(f"Column '{split_column}' not found")
        
        print(f"Found {len(df)} total samples")
        
        # Split data based on the Split column
        df_train = df[df[split_column] == 'train'].copy()
        df_val = df[df[split_column] == 'validation'].copy()
        df_test = df[df[split_column] == 'test'].copy()
        
        # Print statistics
        print("\n" + "="*50)
        print("Dataset Statistics:")
        print("="*50)
        print(f"Train samples: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
        print(f"  - Epilepsy (1): {sum(df_train[label_column] == 1)}")
        print(f"  - PNES (0): {sum(df_train[label_column] == 0)}")
        print(f"Validation samples: {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
        print(f"  - Epilepsy (1): {sum(df_val[label_column] == 1)}")
        print(f"  - PNES (0): {sum(df_val[label_column] == 0)}")
        print(f"Test samples: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
        print(f"  - Epilepsy (1): {sum(df_test[label_column] == 1)}")
        print(f"  - PNES (0): {sum(df_test[label_column] == 0)}")
        print("="*50 + "\n")
        
        # Apply train_ratio stratified sampling to training set if needed
        if train_ratio < 1.0:
            print(f"\nApplying train_ratio={train_ratio} with stratified sampling...")
            print(f"Original training set size: {len(df_train)}")
            
            # Get indices for stratified sampling
            X_train_full = df_train[note_column].values
            y_train_full = df_train[label_column].values
            
            # Use train_test_split with train_size=train_ratio and stratify by labels
            # This ensures the same ratio for each class
            X_train, _, y_train, _ = train_test_split(
                X_train_full,
                y_train_full,
                train_size=train_ratio,
                stratify=y_train_full,
                random_state=42
            )
            
            # Calculate full and split counts
            full_Epilepsy = sum(y_train_full == 1)
            full_PNES = sum(y_train_full == 0)
            split_Epilepsy = sum(y_train == 1)
            split_PNES = sum(y_train == 0)
            
            print(f"Sampled training set size: {len(X_train)}")
            print(f"  - Epilepsy (1): {split_Epilepsy} (ratio: {split_Epilepsy / full_Epilepsy:.4f})")
            print(f"  - PNES (0): {split_PNES} (ratio: {split_PNES / full_PNES:.4f})")
        else:
            # Extract texts and labels without sampling
            X_train = df_train[note_column].values
            y_train = df_train[label_column].values
        
        X_val = df_val[note_column].values
        y_val = df_val[label_column].values
        X_test = df_test[note_column].values
        y_test = df_test[label_column].values
        
        if self.mode == "train":
            # Create datasets
            train_dataset = MedicalNoteDataset(X_train, y_train, self.tokenizer)
            val_dataset = MedicalNoteDataset(X_val, y_val, self.tokenizer)
            
            # Train the model
            self.train(train_dataset, val_dataset)
            print("\nTraining completed!")
            
        elif self.mode == "test":
            # Run inference on test set
            print("\nRunning inference on test set...")
            
            predictions = []
            valid_indices = []
            
            for idx, (df_idx, row) in enumerate(df_test.iterrows()):
                print(f"Processing test sample {idx + 1}/{len(df_test)}")
                
                patient_note = str(row[note_column])
                if pd.isna(patient_note) or patient_note.strip() == '':
                    patient_note = "No clinical note provided"
                
                # if len(patient_note) > 5000:
                #     patient_note = patient_note[:5000]
                
                prediction = self.predict(patient_note)
                
                if prediction is not None:
                    predictions.append(prediction)
                    valid_indices.append(idx)
                    print(f"  Prediction: {prediction}, True label: {row[label_column]}")
                else:
                    print(f"  Skipped (no valid response)")
            
            # Keep only rows with valid predictions
            df_valid = df_test.iloc[valid_indices].copy()
            
            # Create output dataframe
            df_output = pd.DataFrame({
                'Label': df_valid[label_column].values,
                'Prediction': predictions
            })
            
            # Calculate metrics
            accuracy = accuracy_score(df_output['Label'], df_output['Prediction'])
            cm = confusion_matrix(df_output['Label'], df_output['Prediction'])
            
            # Create metrics dataframe
            metrics_data = {
                'Metric': ['Accuracy', 'Confusion Matrix TN', 'Confusion Matrix FP', 
                          'Confusion Matrix FN', 'Confusion Matrix TP'],
                'Value': [accuracy, cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]
            }
            df_metrics = pd.DataFrame(metrics_data)
            
            # Save to Excel
            print(f"\nSaving results to: {output_file}")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df_output.to_excel(writer, sheet_name='Predictions', index=False)
                df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            
            print("Processing complete!")
            print(f"Valid predictions: {len(predictions)}/{len(df_test)}")
            print(f"Epilepsy (1): {sum(predictions)}")
            print(f"PNES (0): {len(predictions) - sum(predictions)}")
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Confusion Matrix:\n{cm}")
            
            return df_output


def main():
    """
    Main function to run training or testing
    
    Usage:
        For training: python script.py --mode train --local_model_path /utilities/models/Qwen2.5-32B-Instruct/
        For testing: python script.py --mode test --local_model_path /utilities/models/Qwen2.5-32B-Instruct/
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaMA QLoRA Fine-tuning for Epilepsy/PNES Classification')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    
    # File paths
    parser.add_argument('--input_file', type=str, default='./MIMIC_merged_text_phenotypes_with_split.xlsx',
                        help='Path to input Excel file')
    parser.add_argument('--output_file', type=str, default='./Llama_qlora_results.xlsx',
                        help='Path to output Excel file')
    parser.add_argument('--note_column', type=str, default='NOTE_TEXT',
                        help='Column name for clinical notes')
    parser.add_argument('--label_column', type=str, default='Label',
                        help='Column name for labels')
    parser.add_argument('--split_column', type=str, default='Split',
                        help='Column name for split information')
    
    # Training hyperparameters
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Ratio of training set to use for model training (0.0-1.0), with stratified sampling')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    
    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.3-70B-Instruct',
                        help='HuggingFace model name')
    parser.add_argument('--output_dir', type=str, default='./llama_qlora_checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--local_model_path', type=str, default="/utilities/models/Qwen2.5-32B-Instruct/",
                        help='Path to local model directory (NEW)')
    
    args = parser.parse_args()
    
    print("="*50)
    print(f"Running in {args.mode.upper()} mode")
    print("="*50)
    
    # Initialize classifier
    classifier = EpilepyPNESClassifier(
        model_name=args.model_name,
        mode=args.mode,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        local_model_path=args.local_model_path
    )
    
    # Process data (train or test based on mode)
    classifier.process_excel_file(
        input_file=args.input_file,
        output_file=args.output_file,
        note_column=args.note_column,
        label_column=args.label_column,
        split_column=args.split_column,
        train_ratio=args.train_ratio
    )


if __name__ == "__main__":
    main()
