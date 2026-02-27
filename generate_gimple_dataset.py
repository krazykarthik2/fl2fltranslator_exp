import os
import subprocess
import glob
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.tools.gimple_parser import parse_gimple

C_DIR = "dataset/samples/c"
OUT_DIR = "dataset/gimple_ir"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

c_files = glob.glob(os.path.join(C_DIR, "*.c"))

for c_file in c_files:
    basename = os.path.basename(c_file)
    name_no_ext = os.path.splitext(basename)[0]
    
    print(f"Processing {basename}...")
    
    # Run GCC
    try:
        # Use shell=True for Windows and full paths for safety
        cmd = ["gcc", "-fdump-tree-gimple", "-c", os.path.abspath(c_file), "-o", f"{name_no_ext}.o"]
        subprocess.run(cmd, check=True, capture_output=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running GCC on {basename}: {e}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'None'}")
        continue
    
    # Find gimple file
    gimple_files = glob.glob(f"{name_no_ext}.c.*t.gimple")
    if not gimple_files:
        print(f"GIMPLE file not found for {basename}")
        continue
    
    gimple_file = gimple_files[0]
    
    # Parse gimple
    with open(gimple_file, "r") as f:
        lines = f.readlines()
    
    try:
        ir = parse_gimple(lines)
        out_path = os.path.join(OUT_DIR, f"{name_no_ext}.ir")
        with open(out_path, "w") as f:
            f.write(ir)
        print(f"  Generated {out_path}")
    except Exception as e:
        print(f"  Error parsing {gimple_file}: {e}")
    
    # Cleanup
    if os.path.exists(f"{name_no_ext}.o"): os.remove(f"{name_no_ext}.o")
    for gf in gimple_files: os.remove(gf)

print("Done.")
