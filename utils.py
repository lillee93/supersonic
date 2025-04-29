import os
import re
import subprocess
import tempfile
import difflib
from difflib import unified_diff
import tempfile
from pathlib import Path


def canonicalize_code(source_code, extension):
    if not extension.startswith('.'):
        if extension.lower() == "c++":
            extension = ".cpp"
        elif extension.lower() == "c":
            extension = ".c"
        else:
            extension = "." + extension

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / ('test' + extension)
        with tmp_file.open('w', encoding='utf-8') as f:
            f.write(source_code)
        try:
            gcc_command = ['gcc', '-fpreprocessed', '-dD', '-E', '-P', str(tmp_file)]
            # print("Running GCC with command:", ' '.join(gcc_command))
            gcc_proc = subprocess.Popen(gcc_command,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            try:
                gcc_out, _= gcc_proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                gcc_proc.kill()
                gcc_out, gcc_err = gcc_proc.communicate()

            if gcc_proc.returncode != 0:
                print("GCC process failed with return code:", gcc_proc.returncode)
                return ""

            if not gcc_out.strip():
                print("GCC output is empty!")
            
            clang_command = ['clang-format', '--style=llvm']
            clang_proc = subprocess.Popen(clang_command,
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            try:
                clang_out, _ = clang_proc.communicate(input=gcc_out, timeout=15)
            except subprocess.TimeoutExpired:
                clang_proc.kill()
                clang_out, _ = clang_out.communicate()

            if clang_proc.returncode != 0:
                print("clang-format process failed with return code:", clang_proc.returncode)
                return ""
            return clang_out.decode('utf-8', errors='ignore')
        except Exception as e:
            print("Exception occurred:", e)
            return ""
        finally:
            for filename in [tmp_file]:
                try:
                    os.unlink(filename)
                except OSError:
                    pass

def apply_diff(original_code, diff_text):
    orig_lines = original_code.splitlines()
    new_lines = orig_lines.copy()
    diff_lines = diff_text.splitlines()
    offset = 0
    i = 0
    print(diff_lines)
    while i < len(diff_lines):
        line = diff_lines[i]

        if '@@' not in line:
            i += 1
            continue

        # a regex that tolerates extra leading chars/spaces/dashes
        header_match = re.search(
            r'-(\d+),?(\d*)\s+\+(\d+),?(\d*)\s*@@',
            line
        )

        if not header_match:
            i += 1
            continue

        orig_start = int(header_match.group(1))
        orig_len = int(header_match.group(2)) if header_match.group(2) else 1
        new_start = int(header_match.group(3))
        new_len = int(header_match.group(4)) if header_match.group(4) else 1
        
        i += 1
        hunk_lines = []
        while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
            hunk_lines.append(diff_lines[i])
            i += 1
        
        # Check if there is at least one context line
        if not any(hl.startswith(' ') for hl in hunk_lines):
            continue

        # Count the number of changed lines
        change_count = sum(1 for hl in hunk_lines if hl.startswith('+') or hl.startswith('-'))
        if change_count > 20:
            continue
        new_hunk = []
        for hl in hunk_lines:
            if hl.startswith(' ') or hl.startswith('+'):
                new_hunk.append(hl[1:])
        
        start_index = orig_start - 1 + offset

        #validate the context lines
        contexts = []
        for index, line in enumerate(hunk_lines):
            if line.startswith(' '):
                contexts.append((index, line[1:]))

        matched = False
        for rel_index, context_text in contexts:
            target_index = start_index + rel_index
            if 0 <= target_index < len(new_lines) and new_lines[target_index] == context_text:
                matched = True
                break

        if not matched:
            continue
        
        new_lines[start_index: start_index + orig_len] = new_hunk
        
        offset += len(new_hunk) - orig_len

    return "\n".join(new_lines)


def compute_diff(original_code, optimized_code, context_lines=1):

    orig_lines = original_code.splitlines()
    opt_lines = optimized_code.splitlines()
    diff_lines = list(unified_diff(orig_lines, opt_lines, fromfile='', tofile='', lineterm=''))

    if len(diff_lines) >= 2 and diff_lines[0].startswith('---') and diff_lines[1].startswith('+++'):
        diff_lines = diff_lines[2:]
    diff_text = "\n".join(diff_lines)
    return diff_text

def write_code_to_file(code, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(code)

def compile_code(code, suffix=""):
    tmp = tempfile.NamedTemporaryFile(suffix=f"{suffix}.cpp", delete=False, mode="w")
    tmp.write(code)
    tmp.flush()
    tmp_name = tmp.name
    tmp.close()
    proc = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-c", tmp_name, "-o", os.devnull],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    os.remove(tmp_name)
    return proc.returncode == 0, proc.stderr

def normalize_code(code):
    unified = code.replace('\r\n', '\n')
    return re.sub(r"\s+", "", unified)

def compute_change_percentage(orig, opt):
    o = normalize_code(orig)
    p = normalize_code(opt)
    ratio = difflib.SequenceMatcher(None, o, p).ratio()
    return (1 - ratio) * 100

def is_identical(orig, opt):
    return orig.strip() == opt.strip()