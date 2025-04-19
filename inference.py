import argparse
import json
from transformers import AutoTokenizer, EncoderDecoderModel
import torch
import utils  # Make sure your utils module contains canonicalize_code() and apply_diff()

def generate_optimized_code(original_code: str, model_dir: str):
    # Load the tokenizer and model (BERT2BERT model based on CodeBERT)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = EncoderDecoderModel.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model.to("cuda")
    
    # Canonicalize the source code (e.g. remove comments and normalize formatting)
    canonical_input = utils.canonicalize_code(original_code, "C++")
    
    # Tokenize the canonical input for the model (using truncation; the data collator will dynamically pad)
    inputs = tokenizer([canonical_input], return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate the output (using greedy decoding by default)
    outputs = model.generate(**inputs, max_new_tokens=512)
    diff_text = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    
    # Apply the predicted diff to obtain the final optimized code.
    optimized_code = utils.apply_diff(canonical_input, diff_text)
    return optimized_code, diff_text

def process_test_dataset(model_dir: str, test_file: str, output_file: str, record_index: int = None):
    results = []
    with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    if record_index is not None:
        # Process only the specified record (0-based indexing).
        try:
            line = lines[record_index]
        except IndexError:
            print(f"Record index {record_index} is out of range. Test file contains {len(lines)} records.")
            return
        record = json.loads(line)
        original_code = record.get("original")
        if not original_code:
            print("Record does not contain an 'original' field.")
            return
        optimized_code, diff_text = generate_optimized_code(original_code, model_dir)
        record["predicted_diff"] = diff_text
        record["optimized_code"] = optimized_code
        results.append(record)
    else:
        # Process all records.
        for idx, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {idx} due to JSONDecodeError: {e}")
                continue
            original_code = record.get("original")
            if not original_code:
                continue
            optimized_code, diff_text = generate_optimized_code(original_code, model_dir)
            record["predicted_diff"] = diff_text
            record["optimized_code"] = optimized_code
            results.append(record)
            print(f"Processed record {idx}:")
            print("Predicted Diff:")
            print(diff_text)
            print("Optimized Code:")
            print(optimized_code)
            print("-" * 40)
    
    # Write the results into the output JSONL file.
    with open(output_file, 'w', encoding='utf-8') as outf:
        for record in results:
            json.dump(record, outf, ensure_ascii=False)
            outf.write("\n")
            
    print(f"Processed {len(results)} record(s). Predictions saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a test JSONL dataset to generate predicted diffs and optimized code, then write results to a file."
    )
    parser.add_argument("--model_dir", required=True, help="Directory of the fine-tuned model")
    parser.add_argument("--test_file", required=True, help="Path to the test dataset JSONL file")
    parser.add_argument("--output_file", required=True, help="File path to save the predictions JSONL file")
    parser.add_argument("--record_index", type=int, default=None, help="Optional: Process only the record at this 0-based index")
    args = parser.parse_args()
    
    process_test_dataset(args.model_dir, args.test_file, args.output_file, args.record_index)

if __name__ == "__main__":
    main()
