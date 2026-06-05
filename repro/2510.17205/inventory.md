# Code Inventory — VisiPruner

## What exists in the official repo

### Core implementation
- ✅ `llava/model/language_model/custom_modeling_llama.py` — VisiPrunerLlamaAttention with full pruning logic
- ✅ `llava/model/language_model/custom_generation.py` — CustomGenerationMixin for accepting pruning_config
- ✅ `llava/model/language_model/llava_llama.py` — LLaVA model that uses VisiPruner attention
- ✅ `llava/serve/cli_pruning.py` — Single-example inference with pruning

### Evaluation scripts (with pruning config)
- ✅ `scripts/v1_5/visiPruner_eval/gqa.sh` — GQA eval with pruning
- ✅ `scripts/v1_5/visiPruner_eval/mme.sh` — MME eval with pruning
- ✅ `scripts/v1_5/visiPruner_eval/textvqa.sh` — TextVQA eval with pruning

### Evaluation modules
- ✅ `llava/eval/model_vqa_loader.py` — General VQA evaluation (used by GQA, MME, TextVQA)
- ✅ `llava/eval/eval_textvqa.py` — TextVQA-specific evaluation
- ✅ `llava/eval/model_vqa.py` — VQA model wrapper

### Standard (non-pruning) eval scripts for comparison
- ✅ `scripts/v1_5/eval/gqa.sh`
- ✅ `scripts/v1_5/eval/mme.sh`
- ✅ `scripts/v1_5/eval/textvqa.sh`
- ✅ `scripts/v1_5/eval/pope.sh`
- ✅ `scripts/v1_5/eval/mmbench.sh`
- ✅ `scripts/v1_5/eval/mmvet.sh`
- ✅ `scripts/v1_5/eval/sqa.sh`
- ✅ `scripts/v1_5/eval/vqav2.sh`
- ✅ `scripts/v1_5/eval/vizwiz.sh`

### Visualization tools
- ✅ Jupyter notebooks for logits lens, attention visualization, L1 norm analysis

### What's missing / needs attention
- ⚠️ Model checkpoints need to be downloaded from HuggingFace
- ⚠️ Evaluation datasets need to be downloaded (following LLaVA instructions)
- ⚠️ No explicit environment.yml — need to install from pyproject.toml
- ⚠️ No explicit results logging — need to capture and compare outputs
