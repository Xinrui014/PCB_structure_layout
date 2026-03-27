#!/usr/bin/env python3
"""Fine-tune Qwen2.5-0.5B on PCB layout generation (ChatML JSONL)."""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse


class PCBLayoutDataset(Dataset):
    """Load JSONL ChatML dataset for causal LM training."""

    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                self.data.append(entry["messages"])

        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]

        # Apply chat template — Qwen2.5 uses ChatML natively
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # For causal LM: labels = input_ids, mask prompt tokens with -100
        # Find where assistant response starts
        labels = input_ids.copy()

        # Mask everything before the assistant's response
        # In ChatML: <|im_start|>assistant\n is the marker
        assistant_marker = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        marker_len = len(assistant_marker)

        # Find last occurrence of assistant marker
        found = -1
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i : i + marker_len] == assistant_marker:
                found = i + marker_len

        if found > 0:
            # Mask all tokens before assistant content
            for i in range(found):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--train_data",
        default="/home/xinrui/projects/data/ti_pcb/layout_data/train.jsonl",
    )
    parser.add_argument(
        "--val_data",
        default="/home/xinrui/projects/data/ti_pcb/layout_data/test.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/xinrui/projects/data/ti_pcb/layout_model/qwen05b_lora",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ["WANDB_PROJECT"] = "pcb_layout"

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register v3 special tokens (SUM_* and ORI_*)
    v3_special_tokens = [
        "[SUM_START]", "[SUM_RES]", "[SUM_CAP]", "[SUM_IND]", "[SUM_CON]",
        "[SUM_DIO]", "[SUM_LED]", "[SUM_SWI]", "[SUM_TRN]", "[SUM_IC]", "[SUM_OSC]",
        "[SUM_END]",
        "[ORI_000]", "[ORI_045]", "[ORI_090]", "[ORI_135]",
        "[ORI_180]", "[ORI_225]", "[ORI_270]", "[ORI_315]",
    ]
    num_added = tokenizer.add_tokens(v3_special_tokens, special_tokens=True)
    print(f"Added {num_added} new special tokens (vocab size: {len(tokenizer)})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # device_map removed for DDP,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = PCBLayoutDataset(args.train_data, tokenizer, args.max_length)
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = PCBLayoutDataset(args.val_data, tokenizer, args.max_length)
    else:
        val_dataset = None
        print("No validation data found, skipping evaluation")

    # Check a sample
    sample = train_dataset[0]
    print(f"Sample input_ids length: {len(sample['input_ids'])}")
    prompt_tokens = sum(1 for l in sample["labels"] if l == -100)
    print(f"Prompt tokens (masked): {prompt_tokens}, Response tokens: {len(sample['labels']) - prompt_tokens}")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.save_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"pcb_layout_{os.path.basename(args.output_dir)}",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
