import torch
import os
import glob

def check_latest(ck_dir):
    files = sorted(glob.glob(os.path.join(ck_dir, "*.pt")), key=os.path.getmtime)
    if not files:
        print(f"No checkpoints in {ck_dir}")
        return
    ckpt_path = files[-1]
    print(f"Checking {ckpt_path}...")
    try:
        # Using weights_only=True might fail if vocab contains custom objects, 
        # but let's try weights_only=False for certainty on keys.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"Keys: {list(ckpt.keys())}")
        print(f"src_vocab: {'Present' if 'src_vocab' in ckpt and ckpt['src_vocab'] is not None else 'Missing'}")
        print(f"tgt_vocab: {'Present' if 'tgt_vocab' in ckpt and ckpt['tgt_vocab'] is not None else 'Missing'}")
        if "src_vocab" in ckpt and ckpt["src_vocab"]:
            print(f"src_vocab size: {len(ckpt['src_vocab'])}")
    except Exception as e:
        print(f"Error checking {ckpt_path}: {e}")

if __name__ == "__main__":
    check_latest("checkpoints/c_to_ir")
    check_latest("checkpoints/ir_to_rust")
