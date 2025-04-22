import csv
import os
import sys

def extract_codenet_pairs(codenet_path):
    pairs = []
    problem_list_file = os.path.join(codenet_path, "metadata", "problem_list.csv")
    if not os.path.isfile(problem_list_file):
        raise FileNotFoundError(f"Cannot find {problem_list_file}")
    allowed_sources = {"AIZU", "AtCoder"}
    allowed_problems = set()
    with open(problem_list_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            problem_id = row[0]
            source = row[2]
            if source in allowed_sources:
                allowed_problems.add(problem_id)
    meta_dir = os.path.join(codenet_path, "metadata")
    for problem_id in allowed_problems:
        meta_file = os.path.join(meta_dir, f"{problem_id}.csv")
        if not os.path.isfile(meta_file):
            continue
        user_subs = {}
        with open(meta_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                sub_id, prob_id, user_id, timestamp, orig_lang, lang, ext, status, runtime, memory, code_size, *_ = row
                if status != "Accepted":
                    continue
                if lang not in ("C", "C++"):
                    continue
                if user_id not in user_subs:
                    user_subs[user_id] = []
                runtime_val = float(runtime) if runtime else float('inf')
                memory_val = float(memory) if memory else float('inf')
                user_subs[user_id].append({
                    "sub_id": sub_id,
                    "lang": lang,
                    "ext": ext,
                    "time": runtime_val,
                    "mem": memory_val,
                    "ts": int(timestamp)
                })
        for user_id, subs in user_subs.items():
            if len(subs) < 2:
                continue
            subs.sort(key=lambda x: x["ts"])
            for i in range(len(subs) - 1):
                prev = subs[i]
                curr = subs[i+1]
                if (curr["time"] < prev["time"]) or (curr["mem"] < prev["mem"]):
                    pairs.append((problem_id, user_id,
                                  prev["sub_id"], prev["lang"], prev["ext"],
                                  curr["sub_id"], curr["lang"], curr["ext"]))
    return pairs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    code_path = sys.argv[1]
    result_pairs = extract_codenet_pairs(code_path)
    print(f"Found {len(result_pairs)} candidate pairs from CodeNet.")
