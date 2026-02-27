import torch
import os

def debug_ckpt(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Keys: {list(ckpt.keys())}")
    if "src_vocab" in ckpt:
        print(f"src_vocab len: {len(ckpt['src_vocab'])}")
        # Print first few items
        print(f"src_vocab sample: {list(ckpt['src_vocab'].items())[:10]}")
    if "tgt_vocab" in ckpt:
        print(f"tgt_vocab len: {len(ckpt['tgt_vocab'])}")
        print(f"tgt_vocab sample: {list(ckpt['tgt_vocab'].items())[:10]}")
    
    # Check model state dict shapes
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        for k in ["seq2seq.encoder.embed.weight", "seq2seq.decoder.embed.weight", "seq2seq.output_proj.weight"]:
            if k in sd:
                print(f"{k} shape: {sd[k].shape}")

if __name__ == "__main__":
    debug_ckpt("checkpoints/c_to_ir/epoch_001_loss_8.8402.pt")
