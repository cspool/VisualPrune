# VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs

- **arXiv**: https://arxiv.org/abs/2510.17205
- **Venue**: EMNLP 2025 Main
- **Code**: https://github.com/EIT-NLP/VisiPruner

## Key Finding: Three-Stage Cross-Modal Interaction

1. **Shallow layers (1-6)**: Recognize task intent. Visual tokens act as passive attention sinks — cross-attention can be merged to a single sink token without performance loss.
2. **Middle layers (7-~24)**: Abrupt cross-modal fusion occurs, driven by a few (avg ~10.3) critical visual tokens.
3. **Deep layers (~24-32)**: Vision tokens are discarded; focus shifts entirely to linguistic refinement. Vision exit layer for LLaVA-v1.5 7B averages 23.9.

## Method: VisiPruner

Training-free pruning framework with three stage-specific strategies:

1. **Shallow pruning**: Layer 1 merges all vision→text cross-attention into a single randomly chosen visual sink token. Layers 2-6 skip all text→vision cross-attention entirely.
2. **Middle pruning**: Value-aware token selection — compute contribution of each visual token via `attn_weights × value_states`, mask each and measure cosine similarity drop. If `cos_sim < layer_threshold (0.995)`, keep token if `L2 distance > tokens_threshold (0.2)`.
3. **Deep exit**: Monitor kept tokens across layers. If they have no measurable impact for 2 consecutive layers, exit vision entirely.

## Key Results — LLaVA-v1.5 7B

| Metric | Dense | VisiPruner | Delta |
|--------|-------|------------|-------|
| GQA | 62.0 | 60.3 | -1.7 |
| SQA-I | 66.8 | 66.7 | -0.1 |
| TextVQA | 58.2 | 55.2 | -3.0 |
| POPE | 85.9 | 84.4 | -1.5 |
| MME-P | 1507.6 | 1428.3 | -79.3 |
| MMBench | 64.3 | 62.0 | -2.3 |
| MMStar | 33.7 | 33.3 | -0.4 |
| **Avg** | **63.8** | **61.9** | **-1.9** |
| FLOPs | 3.82T | 1.76T | -53.9% |

## Baseline Comparison (LLaVA-v1.5 7B, Table 6)

| Method | Vis Attn Reduction | Avg Score |
|--------|-------------------|-----------|
| Full | — | 63.4 |
| PyramidDrop (192) | -86.4% | 61.5 |
| SparseVLM (192) | -86.4% | 62.7 |
| FastV (k=3, r=0.75) | -87.3% | 61.1 |
| **VisiPruner** | **-98.3%** | **61.3** |

## Pruning Configuration

Default hyperparameters (from code):
- `mode`: ["shallow", "middle", "deep"]
- `shallow_mid_layer`: 6 (boundary between shallow and middle)
- `layer_threshold`: 0.995 (cosine similarity threshold for filtering)
- `tokens_threshold`: 0.2 (L2 distance threshold for token retention)

## Model Details

- Base: LLaVA-v1.5 7B (liuhaotian/llava-v1.5-7b)
- Vision encoder: CLIP ViT-L/14 (576 tokens per image)
- LLM: Vicuna-7B-v1.5 (32 layers)
- Vision tokens occupy positions 35 to 35+576=611 (for single image)
- Also evaluated: LLaVA-v1.5 13B, InternVL2.5 8B, QwenVL-v2 7B, MobileVLM-v2 3B

## Evaluation Datasets

- GQA (testdev_balanced)
- MME (Perception + Cognition)
- TextVQA
- POPE
- MMBench
- MMStar
- ScienceQA-IMG
- MM-Vet
