import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/mnt/checkpoints/qwen_lora")
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_instruction(example, tokenizer, labels):
    label_str = ", ".join(labels)
    instruction = (
        f"请判断以下文本的情感类别。可选类别：{label_str}。\n\n"
        f"文本：{example['text']}\n\n类别："
    )
    messages = [
        {"role": "system", "content": "你是一个文本分类专家，只输出类别标签，不输出其他内容。"},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": example["label"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt


def tokenize_function(examples, tokenizer, max_length):
    """对已经格式化好的文本进行 tokenize"""
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    # 对于 causal LM，labels 和 input_ids 相同
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in model_inputs["input_ids"]
    ]
    return model_inputs


def main():
    args = parse_args()
    labels = [l.strip() for l in args.labels.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_data = load_jsonl(args.train_file)
    val_data = load_jsonl(args.val_file)

    # 格式化指令
    train_texts = [format_instruction(d, tokenizer, labels) for d in train_data]
    val_texts = [format_instruction(d, tokenizer, labels) for d in val_data]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    # Tokenize
    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_len),
        batched=True,
        remove_columns=["text"],
    )
    val_ds = val_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_len),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="adamw_torch",
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(f"{args.output_dir}/best")
    tokenizer.save_pretrained(f"{args.output_dir}/best")
    print(f"模型已保存至 {args.output_dir}/best")


if __name__ == "__main__":
    main()
