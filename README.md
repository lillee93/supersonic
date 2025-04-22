# Supersonic (Reproduction)

A minimal guide to build, train, infer and evaluate.

---

## 1. Build dataset  
```bash
python build_dataset.py --codenet_dir --output_dir -max_pairs --workers
```  
Produces `data/{train,val,test}.jsonl`.

---

## 2. Train  
```bash
python train.py --train_file --val_file --output_dir --tokenizer_dir --batch_size
```
---

## 3. Inference  
```bash
python infer.py --model_dir --test_file --output_file --num_beams  --max_records --max_records  --do_sample
```
---

## 4. Evaluate  
```bash
python evaluate.py original_file optimized_file --input-file
```
---

## Key utils (`utils.py`)  
- **canonicalize_code**: strip comments/macros + run clang‑format  
- **compute_diff**: produce diff 
- **apply_diff**: apply diff onto original code