import subprocess
import os
import tempfile
import time
import threading

import psutil

def compile_cpp(source_path, exe_path, extra_flags=None):
    cmd = ["g++", "-O2", source_path, "-o", exe_path]
    if extra_flags:
        cmd[1:1] = extra_flags
    subprocess.run(cmd, check=True)

def run_and_profile(exe_path, stdin_data=None, timeout=None):
    if stdin_data is not None:
        stdin_pipe = subprocess.PIPE
        input_bytes = stdin_data.encode()
    else:
        stdin_pipe = None
        input_bytes = None

    proc = psutil.Popen([exe_path], stdin=stdin_pipe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    peak_rss = 0

    def _monitor():
        nonlocal peak_rss
        try:
            while proc.is_running():
                mem = proc.memory_info().rss
                if mem > peak_rss:
                    peak_rss = mem
                time.sleep(0.005)
        except psutil.NoSuchProcess:
            pass

    monitor_thr = threading.Thread(target=_monitor, daemon=True)
    monitor_thr.start()

    start = time.perf_counter()
    try:
        out, err = proc.communicate(input=input_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    end = time.perf_counter()

    try:
        mem = proc.memory_info().rss
        if mem > peak_rss:
            peak_rss = mem
    except Exception:
        pass

    monitor_thr.join(timeout=0.1)

    runtime = end - start
    return proc.returncode, runtime, peak_rss, out.decode(errors="ignore"), err.decode(errors="ignore")

def profile_cpp(source_path, stdin_data=None, timeout=10):
    base = os.path.splitext(os.path.basename(source_path))[0]
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, base)
        compile_cpp(source_path, exe)
        code, t_sec, mem_bytes, out, err = run_and_profile(exe, stdin_data=stdin_data, timeout=timeout)
        return {
            "return_code": code,
            "time_s": t_sec,
            "peak_memory_mb": (mem_bytes / (1024 * 1024)) if mem_bytes is not None else None,
            "stdout": out,
            "stderr": err,
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="profile a program")
    parser.add_argument("source", help="Path to the file")
    parser.add_argument("--input", "-i", help="Text to feed to STDIN", default=None)
    parser.add_argument("--timeout", "-t", type=float, help="Kill after N seconds", default=10.0)
    args = parser.parse_args()

    results = profile_cpp(args.source, stdin_data=args.input, timeout=args.timeout)

    print(f"Time elapsed:  {results['time_s']:.4f} s")
    if results["peak_memory_mb"] is not None:
        print(f"Peak memory:   {results['peak_memory_mb']:.2f} MB")
    else:
        print("Peak memory:   (could not measure)")
