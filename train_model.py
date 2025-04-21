import os
import argparse
import functools
import datasets
from transformers import (
    AutoTokenizer, 
    EncoderDecoderModel, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
import evaluate
from datasets import load_dataset

def compute_metrics(eval_preds, tokenizer):
    metric = evaluate.load('exact_match')
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {'exact_match': result['exact_match']}

# def eval_metrics(eval_preds, tokenizer):
#     preds, labels = eval_preds
#     # Decode the predictions
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     # Replace pad tokens (-100) in labels with the actual pad token id so we can decode properly
#     labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]

#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
#     # Remove extra whitespace for a clean comparison
#     decoded_preds = [pred.strip() for pred in decoded_preds]
#     decoded_labels = [label.strip() for label in decoded_labels]
    
#     # Compute exact match accuracy
#     correct = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred == label)
#     accuracy = correct / len(decoded_preds) if decoded_preds else 0
#     return {"accuracy": accuracy}
def process_data_to_model_inputs(batch, tokenizer):
  # Tokenize the input and target data
    inputs = tokenizer(batch['original'], padding='max_length', truncation=True, max_length=512)
    outputs = tokenizer(batch['diff'], padding='max_length', truncation=True, max_length=512)

    batch['input_ids'] = inputs.input_ids
    batch['attention_mask'] = inputs.attention_mask
    batch['decoder_attention_mask'] = outputs.attention_mask
    batch['labels'] = outputs.input_ids.copy()

    batch['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch['labels']]

    return batch

def train_model(model_name, train_file, val_file, output_dir, tokenizer_dir, batch_size):

    data_files = {"train": train_file, "validation": val_file}
    raw_datasets = load_dataset("json", data_files=data_files)
    
    # Load the tokenizer from the given model name/path.
    tokenizer = AutoTokenizer.from_pretrained(model_name, users_fast=True)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    
    # Load a BERT2BERT (EncoderDecoder) model using CodeBERT for both encoder and decoder.
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    model.config.max_length = 512
    model.config.early_stopping = True
    model.config.num_beams = 10

    model.config.vocab_size = model.config.encoder.vocab_size
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_dropout = 0.1
    process_data_to_model_inputs_fn = functools.partial(process_data_to_model_inputs, tokenizer=tokenizer)
    # Define the preprocessing function to tokenize both the source code ("original")
    # and the target diff ("diff"), applying truncation up to a maximum length.
    # def preprocess(example):
    #     model_inputs = tokenizer(example["original"], truncation=True, max_length=512)
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(example["diff"], truncation=True, max_length=512)
    #     labels_ids = labels["input_ids"]
    #     # Replace pad token ids with -100 so that they're ignored in the loss computation.
    #     if tokenizer.pad_token_id is not None:
    #         labels_ids = [l if l != tokenizer.pad_token_id else -100 for l in labels_ids]
    #     # model_inputs["attention_mask"] = model_inputs["attention_mask"]

    #     # model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    #     model_inputs["labels"] = labels_ids
    #     return model_inputs

    # tokenized_train = raw_datasets["train"].map(preprocess, remove_columns=raw_datasets["train"].column_names)
    # tokenized_val = raw_datasets["validation"].map(preprocess, remove_columns=raw_datasets["validation"].column_names)
    train_dataset = datasets.load_dataset('json', data_files=train_file, split='train')
    valid_dataset = datasets.load_dataset('json', data_files=val_file, split='train')

    train_dataset = train_dataset.map(
        process_data_to_model_inputs_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=['problem_id', 'original', 'diff']
    )
    train_dataset.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'decoder_attention_mask', 'labels'])

    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=['problem_id', 'original', 'diff']
    )
    valid_dataset.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'decoder_attention_mask', 'labels'])

    training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy='steps',
    do_train=True,
    do_eval=True,
    save_strategy="steps",
    predict_with_generate=True,
    overwrite_output_dir=True,
    greater_is_better=True,
    warmup_steps=500,
    learning_rate=5e-5,
    save_total_limit=5,
    logging_strategy="steps",
    label_smoothing_factor=0,
    fp16=True,
    gradient_accumulation_steps = 4,
    max_steps = 200000,
    eval_steps=2000,
    save_steps=2000,
    logging_steps=50,
    load_best_model_at_end=True,
    logging_dir=os.path.join(output_dir, "logs")
    )
    
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # Bind the tokenizer to our evaluation metric function.
    eval_metrics_fn = functools.partial(compute_metrics, tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer = tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=eval_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(tokenizer_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for source code optimization diffs")
    parser.add_argument("--model_name", default="neulab/codebert-cpp", help="HuggingFace model name or path")
    parser.add_argument("--train_file", default="train.jsonl", help="Path to the training dataset JSONL file")
    parser.add_argument("--val_file", default="val.jsonl", help="Path to the validation dataset JSONL file")
    parser.add_argument("--output_dir", default="supersonic_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--tokenizer_dir", default="supersonic_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    args = parser.parse_args()
    train_model(args.model_name, args.train_file, args.val_file, args.output_dir, args.tokenizer_dir, args.batch_size)


# import os
# import argparse
# import functools
# from transformers import (
#     AutoTokenizer, 
#     EncoderDecoderModel, 
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer, 
#     DataCollatorForSeq2Seq
# )
# from datasets import load_dataset

# def compute_metrics(eval_preds, tokenizer):
#     preds, labels = eval_preds
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     correct = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred.strip() == label.strip())
#     accuracy = correct / len(decoded_preds) if decoded_preds else 0
#     return {"accuracy": accuracy}

# def train_model(model_name, train_file, val_file, output_dir, epochs, batch_size):
#     data_files = {"train": train_file, "validation": val_file}
#     raw_datasets = load_dataset("json", data_files=data_files)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.bos_token is None:
#         tokenizer.bos_token = tokenizer.cls_token or "[CLS]"
#     if tokenizer.eos_token is None:
#         tokenizer.eos_token = tokenizer.sep_token or "[SEP]"
    
#     # Initialize BERT2BERT model using CodeBERT for both encoder and decoder.
#     model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
#     model.config.decoder_start_token_id = tokenizer.bos_token_id
#     model.config.eos_token_id = tokenizer.eos_token_id
#     model.config.pad_token_id = tokenizer.pad_token_id

#     def preprocess(example):
#         model_inputs = tokenizer(example["original"], truncation=True, max_length=512)
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(example["diff"], truncation=True, max_length=512)
#         labels_ids = labels["input_ids"]
#         if tokenizer.pad_token_id is not None:
#             labels_ids = [l if l != tokenizer.pad_token_id else -100 for l in labels_ids]
#         model_inputs["labels"] = labels_ids
#         return model_inputs

#     tokenized_train = raw_datasets["train"].map(preprocess, remove_columns=raw_datasets["train"].column_names)
#     tokenized_val = raw_datasets["validation"].map(preprocess, remove_columns=raw_datasets["validation"].column_names)
    
#     training_args = Seq2SeqTrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         save_strategy="epoch",
#         predict_with_generate=True,
#         overwrite_output_dir=True,
#         metric_for_best_model="accuracy",
#         greater_is_better=True
#     )
    
#     data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
#     compute_metrics_fn = functools.partial(compute_metrics, tokenizer=tokenizer)
    
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train,
#         eval_dataset=tokenized_val,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics_fn
#     )
    
#     trainer.train()
#     trainer.save_model(output_dir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Fine-tune model for source code optimization diffs")
#     parser.add_argument("--model_name", default="microsoft/codebert-base", help="HuggingFace model name or path")
#     parser.add_argument("--train_file", default="train.jsonl", help="Path to training dataset JSONL")
#     parser.add_argument("--val_file", default="val.jsonl", help="Path to validation dataset JSONL")
#     parser.add_argument("--output_dir", default="supersonic_model", help="Directory to save the fine-tuned model")
#     parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
#     args = parser.parse_args()
#     train_model(args.model_name, args.train_file, args.val_file, args.output_dir, args.epochs, args.batch_size)

# import os
# import argparse
# import functools
# from transformers import (
#     AutoTokenizer, 
#     EncoderDecoderModel, 
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer, 
#     DataCollatorForSeq2Seq
# )
# from datasets import load_dataset

# def eval_metrics(eval_preds, tokenizer):
#     preds, labels = eval_preds
#     # Decode the predictions
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     # Replace pad tokens (-100) in labels with the pad token ID for proper decoding
#     labels = [[(l if l != tokenizer.pad_token_id else tokenizer.pad_token_id) for l in label] for label in labels]
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
#     # Remove extraneous whitespace
#     decoded_preds = [pred.strip() for pred in decoded_preds]
#     decoded_labels = [label.strip() for label in decoded_labels]
    
#     # Compute exact match accuracy
#     correct = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred == label)
#     accuracy = correct / len(decoded_preds) if decoded_preds else 0
#     return {"accuracy": accuracy}

# def train_model(model_name, train_file, val_file, output_dir, epochs, batch_size):
#     # Load dataset from JSONL files
#     data_files = {"train": train_file, "validation": val_file}
#     raw_datasets = load_dataset("json", data_files=data_files)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.bos_token is None:
#         tokenizer.bos_token = tokenizer.cls_token or "[CLS]"
#     if tokenizer.eos_token is None:
#         tokenizer.eos_token = tokenizer.sep_token or "[SEP]"
    
#     # Initialize the EncoderDecoderModel (BERT2BERT using CodeBERT)
#     model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
#     model.config.decoder_start_token_id = tokenizer.bos_token_id
#     model.config.eos_token_id = tokenizer.eos_token_id
#     model.config.pad_token_id = tokenizer.pad_token_id

#     def preprocess(example):
#         # Tokenize the source ("original") and the target ("diff")
#         model_inputs = tokenizer(example["original"], truncation=True, max_length=512)
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(example["diff"], truncation=True, max_length=512)
#         labels_ids = labels["input_ids"]
#         if tokenizer.pad_token_id is not None:
#             # Replace pad token IDs with -100 so they are ignored in loss computation
#             labels_ids = [l if l != tokenizer.pad_token_id else -100 for l in labels_ids]
#         model_inputs["labels"] = labels_ids
#         return model_inputs

#     tokenized_train = raw_datasets["train"].map(preprocess, remove_columns=raw_datasets["train"].column_names)
#     tokenized_val = raw_datasets["validation"].map(preprocess, remove_columns=raw_datasets["validation"].column_names)
    
#     training_args = Seq2SeqTrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         save_strategy="epoch",
#         predict_with_generate=True,
#         overwrite_output_dir=True,
#         metric_for_best_model="accuracy",
#         greater_is_better=True
#     )
    
#     data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
#     # Partially bind the tokenizer to our eval_metrics function.
#     eval_metrics_fn = functools.partial(eval_metrics, tokenizer=tokenizer)
    
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train,
#         eval_dataset=tokenized_val,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=eval_metrics_fn
#     )
    
#     trainer.train()
#     trainer.save_model(output_dir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Fine-tune model for source code optimization diffs")
#     parser.add_argument("--model_name", default="microsoft/codebert-base", help="HuggingFace model name or path")
#     parser.add_argument("--train_file", default="train.jsonl", help="Path to training dataset JSONL")
#     parser.add_argument("--val_file", default="val.jsonl", help="Path to validation dataset JSONL")
#     parser.add_argument("--output_dir", default="supersonic_model", help="Directory to save the fine-tuned model")
#     parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
#     args = parser.parse_args()
#     train_model(args.model_name, args.train_file, args.val_file, args.output_dir, args.epochs, args.batch_size)
