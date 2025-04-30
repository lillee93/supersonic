import json
import os
import sys
from utils import write_code_to_file, compile_code, compute_change_percentage, is_identical

def process_records(jsonl_path):
    parent_dir = "code_results"
    os.makedirs(parent_dir, exist_ok=True)
    total = compiled_ok = compile_err = identical = large_change = paired = 0

    with open(jsonl_path, "r") as f:
        for rec_idx, line in enumerate(f, 1):
            rec = json.loads(line)
            problem_id = rec.get("problem_id", f"rec{rec_idx}")
            orig = rec.get("original", "")
            opt = rec.get("optimized_codes", "")[0]
            total += 1

            if is_identical(orig, opt):
                identical += 1

            change_pct = compute_change_percentage(orig, opt)
            if change_pct > 20:
                large_change += 1

            ok, err = compile_code(opt, suffix=f"_{problem_id}")
            if ok:
                compiled_ok += 1
            else:
                compile_err += 1
                print(f"[ERROR] {problem_id} compile failed:\n{err.strip()}\n")

            if ok and not is_identical(orig, opt) and 0 < change_pct <= 20:
                pair_dir = os.path.join(parent_dir, problem_id)
                os.makedirs(pair_dir, exist_ok=True)
                orig = orig.replace('\r\n', '\n')
                write_code_to_file(orig, os.path.join(pair_dir, "original.cpp"))
                write_code_to_file(opt, os.path.join(pair_dir, "optimized.cpp"))
                paired += 1

    print(f"Total records:                                      {total}")
    print(f"Compiled success:                                        {compiled_ok}")
    print(f"Compile errors:                                     {compile_err}")
    print(f"Identical to original:                              {identical}")
    print(f"Pairs with >20% change (skipped writing):           {large_change}")
    print(f"Pairs written : {paired}")

if __name__ == "__main__":
    process_records(sys.argv[1])