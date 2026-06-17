# Gaps Filled — VisiPruner Reproduction

## Issues Found and Fixed

### 1. Device mismatch with multi-GPU 
- **Problem**: Default `device_map="auto"` splits model across GPUs, but custom pruning code doesn't handle multi-device tensors
- **Fix**: Use `device_map="cuda:0"` to keep model on single GPU
- **Paper section**: N/A (implementation detail)

### 2. Editable install requires setuptools>=64
- **Problem**: pyproject.toml build backend doesn't support PEP 660 with old setuptools
- **Fix**: Upgrade setuptools to 82.0.1 before `pip install -e .`

### 3. HF cache directory
- **Problem**: Model downloaded to custom cache needs `HF_HOME` set
- **Fix**: `export HF_HOME=/workspace/VisPrune/models`

## Pruning Hyperparameters (verified from code)

| Parameter | Value | Paper Reference |
|-----------|-------|-----------------|
| mode | ["shallow", "middle", "deep"] | Sec 3.6, 4.4, 5.3 |
| shallow_mid_layer | 6 | Sec 3.6 |
| layer_threshold | 0.995 | Sec 4.4 |
| tokens_threshold | 0.2 | Sec 4.4 |

## What Works
- ✅ Model loads in ~3s on RTX 4090 (fp16, 14.2GB)
- ✅ Inference with pruning produces correct outputs ("Volkswagen")
- ✅ Pruning config propagates through all 32 layers
- ✅ Shallow, middle, and deep pruning modes all active

## What's Needed to Complete

### Evaluation datasets (require user to download):
1. **POPE** (fastest to set up): 
   - COCO val2014 images: `wget http://images.cocodataset.org/zips/val2014.zip`
   - Place in `./playground/data/eval/pope/val2014/`
   
2. **GQA**: Images from https://cs.stanford.edu/people/dorarad/gqa/download.html
   - Also need eval script with fix: https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187

3. **MME**: MME_Benchmark_release_version from official MME repo
   - Also need eval_tool/calculation.py

4. **TextVQA**: Images from https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
