import argparse
import subprocess
import tempfile
import os

def compile_source(source_path, output_exe):
    compiler = "g++" if source_path.endswith((".cpp", ".cc", ".cxx")) else "gcc"
    compile_cmd = [compiler, "-O2", source_path, "-o", output_exe]
    result = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0

def run_program(exe_path, input_data=None):
    cmd = ["/usr/bin/time", "-f", "%e %M", exe_path]
    proc = subprocess.run(cmd, input=input_data.encode('utf-8') if input_data else None,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = proc.stdout
    time_mem = proc.stderr.strip().split()
    time_secs = float(time_mem[0]) if time_mem else None
    mem_kb = float(time_mem[1]) if len(time_mem) > 1 else None
    return output, time_secs, mem_kb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate optimized code against original for correctness and performance")
    parser.add_argument("--orig_file", required=True, help="Path to original source code file")
    parser.add_argument("--opt_file", required=True, help="Path to optimized source code file")
    parser.add_argument("--input_file", help="Path to an input file for running the programs (optional)")
    args = parser.parse_args()
    orig = args.orig_file
    opt = args.opt_file
    inp = None
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            inp = f.read()
    orig_exe = tempfile.NamedTemporaryFile(delete=False).name
    opt_exe = tempfile.NamedTemporaryFile(delete=False).name
    orig_compiled = compile_source(orig, orig_exe)
    opt_compiled = compile_source(opt, opt_exe)
    if not orig_compiled or not opt_compiled:
        print("Compilation failed for", "original" if not orig_compiled else "optimized")
        try: os.unlink(orig_exe)
        except: pass
        try: os.unlink(opt_exe)
        except: pass
        exit(1)
    try:
        orig_out, orig_time, orig_mem = run_program(orig_exe, input_data=inp if inp else "")
        opt_out, opt_time, opt_mem = run_program(opt_exe, input_data=inp if inp else "")
    finally:
        try: os.unlink(orig_exe)
        except: pass
        try: os.unlink(opt_exe)
        except: pass
    outputs_match = (orig_out == opt_out)
    print(f"Output Correctness: {'PASS' if outputs_match else 'FAIL'}")
    print(f"Original Runtime: {orig_time:.3f}s, Memory: {orig_mem:.0f}KB")
    print(f"Optimized Runtime: {opt_time:.3f}s, Memory: {opt_mem:.0f}KB")
    if orig_time and opt_time:
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        print(f"Speedup (orig/opt): {speedup:.2f}x")
    if orig_mem and opt_mem:
        mem_ratio = orig_mem / opt_mem if opt_mem > 0 else float('inf')
        print(f"Memory usage ratio (orig/opt): {mem_ratio:.2f}x")
