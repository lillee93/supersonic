import os
import json
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from tqdm import tqdm
import codenet_extract, utils

def get_codenet_code(codenet_path, problem_id, sub_id, lang, ext):
    code_file = os.path.join(codenet_path, "data", problem_id, lang, f"{sub_id}.{ext}")
    if not os.path.isfile(code_file):
        raise FileNotFoundError(f"Code file {code_file} not found")
    with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def process_candidate(pair_tuple, codenet_path):
    try:
        source, pair = pair_tuple
        problem_id, user_id, sub_old, lang_old, ext_old, sub_new, lang_new, ext_new = pair
    except Exception:
        return None
    try:
        code_old = get_codenet_code(codenet_path, problem_id, sub_old, lang_old, ext_old)
        code_new = get_codenet_code(codenet_path, problem_id, sub_new, lang_new, ext_new)
    except Exception:
        return None

    lang_name = lang_old if lang_old == lang_new else ("C++" if "C++" in (lang_old, lang_new) else "C")
    try:
        canon_old = utils.canonicalize_code(code_old, lang_name)
        canon_new = utils.canonicalize_code(code_new, lang_name)
    except Exception:
        return None
    
    ratio = SequenceMatcher(None, canon_old, canon_new).ratio()
    if ratio < 0.8:
        return None

    diff_text = utils.compute_diff(canon_old, canon_new)

    if diff_text.strip() == "":
        return None

    diff_lines = [line for line in diff_text.splitlines() if line and (line[0] in ('+', '-')) and not line.startswith('+++') and not line.startswith('---')]
    if len(diff_lines) > 20:
        return None

    orig_line_count = canon_old.count('\n') + 1
    if (len(diff_lines) / orig_line_count) > 0.2:
        return None

    return {"problem_id": problem_id, "original": canon_old, "diff": diff_text}

def build_dataset(codenet_path, output_dir, max_pairs=None, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    print("Extracting CodeNet pairs...")
    cn_pairs = codenet_extract.extract_codenet_pairs(codenet_path)
    pairs = [("codenet", p) for p in cn_pairs]
    if max_pairs:
        pairs = pairs[:max_pairs]
    print(f"Total candidate pairs collected from CodeNet: {len(pairs)}")
    processed = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_candidate, pair, codenet_path): pair for pair in pairs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pairs"):
            result = future.result()
            if result is not None:
                processed.append(result)

    print(f"Pairs after filtering: {len(processed)}")
    if not processed:
        print("No valid pairs were found.")
        return

    random.shuffle(processed)
    n = len(processed)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set = processed[:n_train]
    val_set = processed[n_train:n_train + n_val]
    test_set = processed[n_train + n_val:]

    def write_jsonl(data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    write_jsonl(train_set, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(val_set, os.path.join(output_dir, "val.jsonl"))
    write_jsonl(test_set, os.path.join(output_dir, "test.jsonl"))
    print(f"Dataset saved to {output_dir} (train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build dataset from CodeNet submissions")
    parser.add_argument("--codenet_dir", required=True, help="Path to the Project_CodeNet directory")
    parser.add_argument("--output_dir", required=True, help="Directory to save the dataset JSONL files")
    parser.add_argument("--max_pairs", type=int, help="Maximum number of candidate pairs to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()
    build_dataset(args.codenet_dir, args.output_dir, max_pairs=args.max_pairs, num_workers=args.workers)
