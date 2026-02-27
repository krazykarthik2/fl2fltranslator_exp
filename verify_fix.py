
import os
import torch
import torch.nn as nn
from src.training.train_c_to_ir import Trainer, TrainingConfig
from src.model.c_to_ir_model import CToIRModel
from src.model.transformer import TransformerConfig

def test_save_load():
    config = TrainingConfig(save_dir="tmp_checkpoints")
    trainer = Trainer(config)
    
    # Mock vocab
    trainer.src_vocab = {"a": 0, "b": 1}
    trainer.tgt_vocab = {"x": 0, "y": 1}
    
    # Save
    trainer.save_checkpoint(1, 0.5)
    
    # Check if file exists
    ckpt_path = os.path.join(config.save_dir, "epoch_001_loss_0.5000.pt")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created"
    
    # Load into a new trainer
    trainer2 = Trainer(config)
    trainer2.load_checkpoint(ckpt_path)
    
    # Verify vocab
    assert hasattr(trainer2, "src_vocab"), "src_vocab not loaded"
    assert hasattr(trainer2, "tgt_vocab"), "tgt_vocab not loaded"
    assert trainer2.src_vocab == trainer.src_vocab, "src_vocab mismatch"
    assert trainer2.tgt_vocab == trainer.tgt_vocab, "tgt_vocab mismatch"
    print("Verification successful: Vocabulary saved and loaded correctly!")

if __name__ == "__main__":
    if not os.path.exists("tmp_checkpoints"):
        os.makedirs("tmp_checkpoints")
    try:
        test_save_load()
    finally:
        # Cleanup
        if os.path.exists("tmp_checkpoints"):
            import shutil
            shutil.rmtree("tmp_checkpoints")
