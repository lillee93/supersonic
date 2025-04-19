import os
import argparse
import functools

import datasets
import evaluate
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)


def load_splits(train_path, val_path):
    ds = datasets.load_dataset("json", data_files={"train": train_path, "validation": val_path})
    return ds["train"], ds["validation"]


def preprocess_batch(batch, tokenizer, max_len):
    src = tokenizer(batch["original"], padding="max_length", truncation=True, max_length=max_len)
    with tokenizer.as_target_tokenizer():
        tgt = tokenizer(batch["diff"], padding="max_length", truncation=True, max_length=max_len)
    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in tgt["input_ids"]
    ]
    return { 
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "decoder_attention_mask": tgt["attention_mask"],
        "labels": labels,
    }


def exact_match_metric(pred_labels, tokenizer):
    metric = evaluate.load("exact_match")
    preds, labels = pred_labels
    if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    import numpy as np
    lab_arr = np.array(labels)
    lab_arr[lab_arr == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(lab_arr, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    res = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"exact_match": res["exact_match"]}


def train(train_ds, valid_ds, out_dir, tok_dir,model_id="neulab/codebert-cpp",max_len=512, bs=4, lr=5e-5,
    steps=20000, eval_steps=1000, grad_acc=4):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_id, model_id)
    tok.bos_token = tok.cls_token
    tok.eos_token = tok.sep_token
    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.max_length = max_len
    model.config.num_beams = 10
    model.config.early_stopping = True
    model.config.vocab_size = model.config.encoder.vocab_size

    fn = functools.partial(preprocess_batch, tokenizer=tok, max_len=max_len)
    tr = train_ds.map(fn, batched=True, remove_columns=train_ds.column_names)
    va = valid_ds.map(fn, batched=True, remove_columns=valid_ds.column_names)
    tr.set_format(type="torch", columns=["input_ids","attention_mask","decoder_attention_mask","labels"])
    va.set_format(type="torch", columns=["input_ids","attention_mask","decoder_attention_mask","labels"])

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=grad_acc,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        max_steps=steps,
        warmup_steps=500,
        learning_rate=lr,
        predict_with_generate=True,
        fp16=True,
        logging_strategy="steps",
        logging_steps=50,
        logging_dir=os.path.join(out_dir, "logs"),
    )

    collate = DataCollatorForSeq2Seq(tok, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tr,
        eval_dataset=va,
        tokenizer=tok,
        data_collator=collate,
        compute_metrics=functools.partial(exact_match_metric, tokenizer=tok),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(tok_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file",   required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--tokenizer_dir",    required=True)
    p.add_argument("--max_len",   type=int,   default=512)
    p.add_argument("--batch_size",type=int,   default=4)
    p.add_argument("--lr",        type=float, default=5e-5)
    p.add_argument("--steps",     type=int,   default=20000)
    p.add_argument("--eval_steps",type=int,   default=2000)
    p.add_argument("--grad_acc",  type=int,   default=4)
    args = p.parse_args()

    tr, va = load_splits(args.train_file, args.val_file)
    train(
        train_ds=tr,
        valid_ds=va,
        out_dir=args.out_dir,
        tok_dir = args.tokenizer_dir,
        max_len=args.max_len,
        bs=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        eval_steps=args.eval_steps,
        grad_acc=args.grad_acc,
    )


if __name__ == "__main__":
    main()
