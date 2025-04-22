import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
import psutil


def compile_source(source_path: str, exe_path: str, extra_flags=None):

    ext = os.path.splitext(source_path)[1].lower()
    driver = "gcc" if ext == ".c" else "g++"
    cmd = [driver, "-O2", source_path, "-o", exe_path]
    if extra_flags:
        cmd[1:1] = extra_flags

    if driver.endswith("g++") and sys.platform.startswith("win"):
        cmd += ["-Wl,-subsystem,console"]
    subprocess.run(cmd, check=True)

def run_and_profile(exe_path, stdin_data = None):

    stdin_pipe = subprocess.PIPE if stdin_data is not None else None
    input_bytes = stdin_data.encode() if stdin_data is not None else None

    proc = psutil.Popen([exe_path], stdin=stdin_pipe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak_rss = 0

    def monitor_mem():
        nonlocal peak_rss
        try:
            while proc.is_running():
                rss = proc.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
                time.sleep(0.005)
        except psutil.NoSuchProcess:
            pass

    mon = threading.Thread(target=monitor_mem, daemon=True)
    mon.start()

    start = time.perf_counter()
    try:
        out, err = proc.communicate(input=input_bytes, timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    end = time.perf_counter()

    try:
        rss = proc.memory_info().rss
        if rss > peak_rss:
            peak_rss = rss
    except Exception:
        pass

    mon.join(timeout=0.1)
    runtime = end - start
    return proc.returncode, runtime, peak_rss, out.decode(errors="ignore"), err.decode(errors="ignore")

def profile_code(source_path, stdin_data = None):

    base = os.path.splitext(os.path.basename(source_path))[0]
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, base)
        compile_source(source_path, exe)
        code, t, mem, out, err = run_and_profile(exe, stdin_data=stdin_data)
        return {
            "return_code": code,
            "time_s": t,
            "peak_memory_mb": mem / (1024 * 1024),
            "stdout": out,
            "stderr": err,
        }

def compare_versions(original, optimized, stdin_data = None):
    r1 = profile_code(original, stdin_data)
    r2 = profile_code(optimized, stdin_data)

    if r1["return_code"] != 0:
        print(f"Error: {original} exited with code {r1['return_code']}", file=sys.stderr)
    if r2["return_code"] != 0:
        print(f"Error: {optimized} exited with code {r2['return_code']}", file=sys.stderr)

    print()
    print(f"{'Program':<30}{'Time (s)':>10}{'Peak Mem (MB)':>15}")
    print("-"*55)
    print(f"{os.path.basename(original):<30}{r1['time_s']:10.4f}{r1['peak_memory_mb']:15.2f}")
    print(f"{os.path.basename(optimized):<30}{r2['time_s']:10.4f}{r2['peak_memory_mb']:15.2f}")
    print()

    if r1["time_s"] and r2["time_s"] > 0:
        speedup = r1["time_s"] / r2["time_s"]
        print(f"Speedup ({os.path.basename(original)}/{os.path.basename(optimized)}): {speedup:.2f}×")
    if r1["peak_memory_mb"] and r2["peak_memory_mb"] > 0:
        mem_ratio = r1["peak_memory_mb"] / r2["peak_memory_mb"]
        print(f"Memory ratio ({os.path.basename(original)}/{os.path.basename(optimized)}): {mem_ratio:.2f}×")

def main():
    parser = argparse.ArgumentParser(
        description="Compile, run and compare two C/C++ programs on the same input"
    )
    parser.add_argument("original", help="Path to the original file")
    parser.add_argument("optimized", help="Path to the optimized file")
    parser.add_argument("--input-file", "-i", default=None)
    args = parser.parse_args()

    inp = None
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8', errors='ignore') as f:
                inp = f.read()
        except FileNotFoundError:
            pass

    compare_versions(args.original, args.optimized, stdin_data=inp)

if __name__ == "__main__":
    main()
