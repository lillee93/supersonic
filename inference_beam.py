import argparse
import json
import torch
from transformers import AutoTokenizer, EncoderDecoderModel
import utils

def generate_optimized_codes(original_code: str, model_dir: str, num_beams: int = 5, 
                               num_return_sequences: int = 3, do_sample: bool = False):

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = EncoderDecoderModel.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model.to("cuda")
    

    canonical_input = utils.canonicalize_code(original_code, "C++")
    
    inputs = tokenizer([canonical_input], return_tensors="pt", truncation=True, 
                         max_length=512, padding="max_length")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=num_beams,
        early_stopping=True,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,      
        top_k=50,                 
        top_p=0.95,              
        temperature=0.8           
    )
    
    candidate_diffs = [tokenizer.decode(output.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                       for output in outputs]
    candidate_optimized = [utils.apply_diff(canonical_input, diff_text) for diff_text in candidate_diffs]
    return candidate_diffs, candidate_optimized

def process_test_dataset(model_dir: str, test_file: str, output_file: str, max_records: int = None,
                         num_beams: int = 5, num_return_sequences: int = 3, do_sample: bool = False):
    results = []
    with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            if max_records is not None and idx >= max_records:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping record {idx} due to JSONDecodeError: {e}")
                continue
            original_code = record.get("original")
            if not original_code:
                continue
            candidate_diffs, candidate_optimized = generate_optimized_codes(
                original_code, model_dir,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample
            )
            record["predicted_diffs"] = candidate_diffs
            record["optimized_codes"] = candidate_optimized
            results.append(record)
            print(f"Record {idx}:")
            for i, diff in enumerate(candidate_diffs):
                print(f"  Candidate {i} Diff:")
                print(diff)
                print("  ---")
            print("=" * 40)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record in results:
            json.dump(record, out_f, ensure_ascii=False)
            out_f.write("\n")
    print(f"Processed {len(results)} record(s). Predictions saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a test JSONL dataset with beam search (and optional sampling) to produce multiple candidate diffs and optimized code outputs."
    )
    parser.add_argument("--model_dir", required=True, help="Directory of the fine-tuned model")
    parser.add_argument("--test_file", required=True, help="Path to the test dataset JSONL file")
    parser.add_argument("--output_file", required=True, help="File path to save the predictions JSONL file")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--num_return_sequences", type=int, default=3, help="Number of candidate sequences to return")
    parser.add_argument("--max_records", type=int, default=None, help="Maximum number of records to process")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling to generate diverse outputs")
    args = parser.parse_args()
    
    process_test_dataset(
        args.model_dir, args.test_file, args.output_file, args.max_records,
        num_beams=args.num_beams, num_return_sequences=args.num_return_sequences, do_sample=args.do_sample
    )

if __name__ == "__main__":
    main()
