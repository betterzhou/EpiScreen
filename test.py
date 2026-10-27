import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
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
Use your knowledge of the clinical characteristics and typical presentations of Epilepsy and PNES to make the classification.

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
    def __init__(self, model_name="/utilities/models/Qwen2.5-32B-Instruct/", 
                 mode="train", 
                 learning_rate=2e-4,
                 num_epochs=3,
                 batch_size=1,
                 gradient_accumulation_steps=8,
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.05,
                 output_dir="./llama_qlora_checkpoints"):
        """
        Initialize the classifier
        
        Args:
            model_name: Local path to model directory
            mode: "train" or "test"
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            gradient_accumulation_steps: Steps to accumulate gradients
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
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
        
        print(f"Mode: {mode}")
        print(f"Loading model from local path: {model_name}")
        
        # Configure 4-bit quantization for QLoRA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer from local path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model with quantization from local path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
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
        return f""" 
        Based on the patient's clinical note, classify if the patient has Epilepsy or PNES (Psychogenic Non-Epileptic Seizures).
Use your knowledge of the clinical characteristics and typical presentations of Epilepsy and PNES to make the classification.

If the prediction is Epilepsy, output 'Epilepsy'. If the prediction is PNES, output 'PNES'.
Output either 'Epilepsy' or 'PNES', nothing else.

Patient's clinical note: {patient_note}

Answer:"""
    
    def predict_probability(self, patient_note: str):
        """
        Predict probability using token logits
        
        Args:
            patient_note: Clinical note text
            
        Returns:
            Probability of Epilepsy (float between 0 and 1)
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
                outputs = self.model(inputs)
                logits = outputs.logits[:, -1, :]  # next-token logits (batch_size, vocab_size)
            
            epilepsy_candidates = [
                self.tokenizer.encode("Epilepsy", add_special_tokens=False)[0],
                self.tokenizer.encode(" Epilepsy", add_special_tokens=False)[0]
            ]
            pnes_candidates = [
                self.tokenizer.encode("PNES", add_special_tokens=False)[0],
                self.tokenizer.encode(" PNES", add_special_tokens=False)[0]
            ]
            
            # Use the token with highest logit
            epilepsy_logits = [logits[0, tid].item() for tid in epilepsy_candidates]
            pnes_logits = [logits[0, tid].item() for tid in pnes_candidates]
            
            logit_epilepsy = max(epilepsy_logits)
            logit_pnes = max(pnes_logits)
            
            # Use log-softmax for numerical stability
            logits_tensor = torch.tensor([logit_epilepsy, logit_pnes])
            log_probs = F.log_softmax(logits_tensor, dim=0)
            probs = torch.exp(log_probs)
            
            # P(Epilepsy)
            prob_epilepsy = probs[0].item()
            
            return prob_epilepsy
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def process_excel_file(self, input_file: str, output_file: str, 
                          note_column: str = 'NOTE_TEXT', 
                          label_column: str = 'Label',
                          split_column: str = 'Split'):
        """
        Process Excel file for training or testing
        
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
        
        # Use Split column to get train/val/test sets
        train_mask = df[split_column] == 'train'
        val_mask = df[split_column] == 'validation'
        test_mask = df[split_column] == 'test'
        
        X_train = df.loc[train_mask, note_column].values
        y_train = df.loc[train_mask, label_column].values
        idx_train = df[train_mask].index.tolist()
        
        X_val = df.loc[val_mask, note_column].values
        y_val = df.loc[val_mask, label_column].values
        idx_val = df[val_mask].index.tolist()
        
        X_test = df.loc[test_mask, note_column].values
        y_test = df.loc[test_mask, label_column].values
        idx_test = df[test_mask].index.tolist()
        
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        if self.mode == "train":
            # Create datasets
            train_dataset = MedicalNoteDataset(X_train, y_train, self.tokenizer)
            val_dataset = MedicalNoteDataset(X_val, y_val, self.tokenizer)
            
            # Train the model
            self.train(train_dataset, val_dataset)
            print("\nTraining completed!")
            
        elif self.mode == "test":
            # Run inference on test set using token logits
            print("\nRunning inference on test set using token logits...")
            df_test = df.iloc[idx_test].copy()
            
            results = []
            
            for idx, (df_idx, row) in enumerate(df_test.iterrows()):
                print(f"Processing test sample {idx + 1}/{len(df_test)}")
                
                patient_note = str(row[note_column])
                if pd.isna(patient_note) or patient_note.strip() == '':
                    patient_note = "No clinical note provided"
                
                # Get probability using token logits
                prob_epilepsy = self.predict_probability(patient_note)
                
                if prob_epilepsy is not None:
                    result_row = {
                        'Label': row[label_column],
                        'Probability': prob_epilepsy
                    }
                    results.append(result_row)
                    
                    print(f"  Probability (Epilepsy): {prob_epilepsy:.4f}, True label: {row[label_column]}")
                else:
                    print(f"  Skipped (prediction failed)")
            
            # Create output dataframe
            df_output = pd.DataFrame(results)
            
            # Calculate AUC
            auc_score = roc_auc_score(df_output['Label'], df_output['Probability'])
            
            # Create metrics dataframe
            df_metrics = pd.DataFrame({
                'Metric': ['AUC Score'],
                'Value': [auc_score]
            })
            
            # Save to Excel
            print(f"\nSaving results to: {output_file}")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df_output.to_excel(writer, sheet_name='Predictions', index=False)
                df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            
            print("Processing complete!")
            print(f"Total valid samples: {len(results)}")
            print(f"AUC Score: {auc_score:.4f}")
            
            return df_output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaMA QLoRA Fine-tuning for Epilepsy/PNES Classification')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    
    # File paths
    parser.add_argument('--input_file', type=str, default='./MIMIC.xlsx',
                        help='Path to input Excel file')
    parser.add_argument('--output_file', type=str, default='./Llama_qlora_results.xlsx',
                        help='Path to output Excel file')
    parser.add_argument('--note_column', type=str, default='NOTE_TEXT',
                        help='Column name for clinical notes')
    parser.add_argument('--label_column', type=str, default='Label',
                        help='Column name for labels')
    
    # Training hyperparameters
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
    parser.add_argument('--model_name', type=str, default='/utilities/models/Qwen2.5-32B-Instruct/',
                        help='Local path to model directory')
    parser.add_argument('--output_dir', type=str, default='./llama_qlora_checkpoints',
                        help='Output directory for checkpoints')
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
        output_dir=args.output_dir
    )
    
    # Process data (train or test based on mode)
    classifier.process_excel_file(
        input_file=args.input_file,
        output_file=args.output_file,
        note_column=args.note_column,
        label_column=args.label_column
    )


if __name__ == "__main__":
    main()

