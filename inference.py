import argparse
import json
import torch
from transformers import AutoTokenizer, EncoderDecoderModel
import utils

def generate_optimized_codes(original_code, model_dir,tokenizer_dir,use_beam = False, beam_size = 5, 
                             num_return_sequences = 1,do_sample = False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    model = EncoderDecoderModel.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    canon = utils.canonicalize_code(original_code, "C++")
    inputs = tokenizer(
        [canon],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    if use_beam:
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=beam_size,
            early_stopping=True,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=1,
            early_stopping=True,
            num_return_sequences=1,
            do_sample=False
        )

    diffs = [
        tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for o in outputs
    ]
    patched = [utils.apply_diff(canon, d) for d in diffs]
    return diffs, patched


def process_test_dataset( model_dir, tokenizer_dir, test_file, output_file, max_records, use_beam = False,
                        beam_size = 5, num_return_sequences = 1, do_sample = False):
    results = []
    with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if max_records is not None and idx >= max_records:
                break
            rec = json.loads(line)
            orig = rec.get("original")
            if not orig:
                continue

            diffs, patched = generate_optimized_codes(
                orig, model_dir, tokenizer_dir,
                use_beam, beam_size, num_return_sequences, do_sample
            )

            rec["predicted_diffs"] = diffs
            rec["optimized_codes"]   = patched
            results.append(rec)

            print(f"\n=== Record {idx} ===")
            for i, d in enumerate(diffs):
                print(f"[{i}] DIFF:\n{d}\n----")

    with open(output_file, "w", encoding="utf-8") as out:
        for rec in results:
            json.dump(rec, out, ensure_ascii=False)
            out.write("\n")
    print(f"\nProcessed {len(results)} records → {output_file}")


def main():
    p = argparse.ArgumentParser(
        description="Inference with optional beam vs greedy; separate model & tokenizer dirs"
    )
    p.add_argument("--model_dir",     required=True, help="Path to the fine‑tuned model")
    p.add_argument("--tokenizer_dir", required=True, help="Path to the tokenizer folder")
    p.add_argument("--test_file",     required=True, help="Input jsonl")
    p.add_argument("--output_file",   required=True, help="Output jsonl")
    p.add_argument("--max_records",   type=int, default=None)
    p.add_argument("--use_beam",      action="store_true")
    p.add_argument("--beam_size",     type=int, default=5)
    p.add_argument("--num_return_sequences", type=int, default=1)
    p.add_argument("--do_sample",     action="store_true")

    args = p.parse_args()
    process_test_dataset(
        args.model_dir,
        args.tokenizer_dir,
        args.test_file,
        args.output_file,
        max_records=args.max_records,
        use_beam=args.use_beam,
        beam_size=args.beam_size,
        num_return_sequences=args.num_return_sequences,
        do_sample=args.do_sample
    )

if __name__ == "__main__":
    main()
