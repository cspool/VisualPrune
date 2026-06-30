#!/usr/bin/env python3
"""Torch reconstruction scaffold for input2_layer18, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input2_layer18'
ORIGINAL_DIMS = json.loads(r"""{
  "ffn": 11008,
  "head_dim": 128,
  "heads": 32,
  "hidden": 4096,
  "kv_len": 625,
  "q_len": 1
}""")
DISPATCH_FEATURES = json.loads(r"""{
  "expected_stages": [
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "kv_cache_concat",
    "attention",
    "attention_output",
    "mlp"
  ],
  "has_attention": true,
  "has_cache_concat": true,
  "has_mlp": true,
  "has_rope": true,
  "kv_len": 625,
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 4,
    "div.Tensor": 1,
    "dropout.default": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 2,
    "item.default": 2,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 9,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 2,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 2,
    "view.default": 3
  },
  "phase": "decode",
  "prune_probe_kind": null,
  "q_len": 1,
  "role": "decode_prune_effect",
  "token_state": "full_cache",
  "visual_adjust_kind": null
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00002475"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002476"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002477"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002478"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002479"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002480"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00002492",
  "t00002497"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002498"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00002499",
  "t00002504"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002505"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00002506"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 625], dtype=float16",
"output_tensor_ids": [
  "t00002508"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00002498",
  "t00002508"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 625], dtype=float16",
"output_tensor_ids": [
  "t00002509"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00002509"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 1, 625], dtype=float16",
"output_tensor_ids": [
  "t00002510"
]
      },
      {
"event_op_index": 52,
"input_tensor_ids": [
  "t00002510",
  "t00002511"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 625], dtype=float16",
"output_tensor_ids": [
  "t00002512"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00002512"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 1, 625], dtype=float32",
"output_tensor_ids": [
  "t00002513"
]
      },
      {
"event_op_index": 55,
"input_tensor_ids": [
  "t00002514"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 1, 625], dtype=float16",
"output_tensor_ids": [
  "t00002514"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00002514",
  "t00002507"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002515"
]
      }
    ],
    "stage": "attention",
    "summary": "Attention evidence is q @ k^T, scale/div, mask add, softmax, and dropout over q_len x kv_len scores."
  },
  "attention_output": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00002514",
  "t00002507"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002515"
]
      },
      {
"event_op_index": 58,
"input_tensor_ids": [
  "t00002516"
],
"op_name": "reshape.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002517"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00002517",
  "t00001432"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002519"
]
      },
      {
"event_op_index": 62,
"input_tensor_ids": [
  "t00002463",
  "t00002519"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002520"
]
      },
      {
"event_op_index": 75,
"input_tensor_ids": [
  "t00002532",
  "t00001450"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002533"
]
      },
      {
"event_op_index": 76,
"input_tensor_ids": [
  "t00002520",
  "t00002533"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002534"
]
      }
    ],
    "stage": "attention_output",
    "summary": "Attention-output evidence is attn @ V, transpose/reshape, output projection, and residual add."
  },
  "input_rmsnorm": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 1,
"input_tensor_ids": [
  "t00002463"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002464"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00002464"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002465"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00002465"
],
"op_name": "mean.dim",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002466"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00002466"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002467"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00002467"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002468"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00002464",
  "t00002468"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002469"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00002469"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002470"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00001354",
  "t00002470"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002471"
]
      }
    ],
    "stage": "input_rmsnorm",
    "summary": "RMSNorm evidence is the initial cast, square, mean, eps-add, rsqrt, and weight multiply sequence."
  },
  "kv_cache_concat": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00001393",
  "t00002505"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 625, 128], dtype=float16",
"output_tensor_ids": [
  "t00002506"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00001367",
  "t00002480"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 625, 128], dtype=float16",
"output_tensor_ids": [
  "t00002507"
]
      }
    ],
    "stage": "kv_cache_concat",
    "summary": "Decode cache evidence is K/V cat outputs whose sequence dimension equals dispatch kv_len."
  },
  "mlp": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 54,
"input_tensor_ids": [
  "t00002513"
],
"op_name": "to.dtype",
"output": "shape=[1, 32, 1, 625], dtype=float16",
"output_tensor_ids": [
  "t00002514"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00002517",
  "t00001432"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002519"
]
      },
      {
"event_op_index": 62,
"input_tensor_ids": [
  "t00002463",
  "t00002519"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002520"
]
      },
      {
"event_op_index": 63,
"input_tensor_ids": [
  "t00002520"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002521"
]
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00002521"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002522"
]
      },
      {
"event_op_index": 65,
"input_tensor_ids": [
  "t00002522"
],
"op_name": "mean.dim",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002523"
]
      },
      {
"event_op_index": 66,
"input_tensor_ids": [
  "t00002523"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002524"
]
      },
      {
"event_op_index": 67,
"input_tensor_ids": [
  "t00002524"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002525"
]
      },
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00002521",
  "t00002525"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002526"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00002526"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002527"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00001442",
  "t00002527"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002528"
]
      },
      {
"event_op_index": 71,
"input_tensor_ids": [
  "t00002528",
  "t00001444"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002529"
]
      },
      {
"event_op_index": 72,
"input_tensor_ids": [
  "t00002529"
],
"op_name": "silu.default",
"output": "shape=[1, 1, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002530"
]
      },
      {
"event_op_index": 73,
"input_tensor_ids": [
  "t00002528",
  "t00001447"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002531"
]
      }
    ],
    "stage": "mlp",
    "summary": "MLP evidence is post-attention RMSNorm, gate/up linear, SiLU, gated product, down linear, residual add."
  },
  "qkv_projection": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 9,
"input_tensor_ids": [
  "t00002471",
  "t00001356"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002472"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00002471",
  "t00001358"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002473"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00002471",
  "t00001360"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002474"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00002472"
],
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002475"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00002475"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002476"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00002473"
],
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002477"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002477"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002478"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00002474"
],
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002479"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002479"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002480"
]
      }
    ],
    "stage": "qkv_projection",
    "summary": "Q/K/V evidence is three hidden-size linear projections followed by view/transpose head split."
  },
  "rope": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 20,
"input_tensor_ids": [
  "t00002483"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002484"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00001372"
],
"op_name": "slice.Tensor",
"output": "shape=[625, 128], dtype=float16",
"output_tensor_ids": [
  "t00002486"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00001374"
],
"op_name": "slice.Tensor",
"output": "shape=[625, 128], dtype=float16",
"output_tensor_ids": [
  "t00002487"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00002486",
  "t00002481"
],
"op_name": "index.Tensor",
"output": "shape=[1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002488"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00002488"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002489"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00002487",
  "t00002481"
],
"op_name": "index.Tensor",
"output": "shape=[1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002490"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00002490"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002491"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00002476",
  "t00002489"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002492"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00002476"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 1, 64], dtype=float16",
"output_tensor_ids": [
  "t00002493"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00002476"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 1, 64], dtype=float16",
"output_tensor_ids": [
  "t00002494"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00002494"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 1, 64], dtype=float16",
"output_tensor_ids": [
  "t00002495"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00002495",
  "t00002493"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002496"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00002496",
  "t00002491"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002497"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00002492",
  "t00002497"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002498"
]
      }
    ],
    "stage": "rope",
    "summary": "RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add."
  }
}""")
SUMMARY = json.loads(r"""{
  "dispatch_features": {
    "expected_stages": [
      "input_rmsnorm",
      "qkv_projection",
      "rope",
      "kv_cache_concat",
      "attention",
      "attention_output",
      "mlp"
    ],
    "has_attention": true,
    "has_cache_concat": true,
    "has_mlp": true,
    "has_rope": true,
    "kv_len": 625,
    "op_counts": {
      "add.Tensor": 8,
      "cat.default": 4,
      "div.Tensor": 1,
      "dropout.default": 1,
      "gt.Scalar": 2,
      "index.Tensor": 2,
      "is_nonzero.default": 2,
      "item.default": 2,
      "linear.default": 7,
      "matmul.default": 2,
      "mean.dim": 2,
      "mul.Tensor": 9,
      "neg.default": 2,
      "pow.Tensor_Scalar": 2,
      "reshape.default": 1,
      "rsqrt.default": 2,
      "select.int": 2,
      "silu.default": 1,
      "slice.Tensor": 6,
      "softmax.int": 1,
      "to.dtype": 7,
      "transpose.int": 5,
      "unsqueeze.default": 2,
      "view.default": 3
    },
    "phase": "decode",
    "prune_probe_kind": null,
    "q_len": 1,
    "role": "decode_prune_effect",
    "token_state": "full_cache",
    "visual_adjust_kind": null
  },
  "dispatch_op_coverage": {
    "covered_op_count": 76,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 76
  },
  "event_id": "input2_layer18",
  "input_id": 2,
  "kv_len": 625,
  "layer_id": 18,
  "module_split": [
    {
      "event_op_indices": [
1,
2,
3,
4,
5,
6,
7,
8
      ],
      "external_input_tensor_ids": [
"t00002463",
"t00001354"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00002463",
"t00002464",
"t00002465",
"t00002466",
"t00002467",
"t00002468",
"t00002469",
"t00001354",
"t00002470"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00002471"
      ],
      "module_path": "model.layers.18.input_layernorm",
      "module_relative_path": "input_layernorm",
      "module_type": "LlamaRMSNorm",
      "op_count": 8,
      "op_counts": {
"add.Tensor": 1,
"mean.dim": 1,
"mul.Tensor": 2,
"pow.Tensor_Scalar": 1,
"rsqrt.default": 1,
"to.dtype": 2
      },
      "output_tensor_ids": [
"t00002464",
"t00002465",
"t00002466",
"t00002467",
"t00002468",
"t00002469",
"t00002470",
"t00002471"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00002471",
"t00001356"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00002471",
"t00001356"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002472"
      ],
      "module_path": "model.layers.18.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002472"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00002471",
"t00001358"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00002471",
"t00001358"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002473"
      ],
      "module_path": "model.layers.18.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002473"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00002471",
"t00001360"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00002471",
"t00001360"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002474"
      ],
      "module_path": "model.layers.18.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002474"
      ]
    },
    {
      "event_op_indices": [
12,
13,
14,
15,
16,
17,
18,
19,
20,
29,
30,
31,
32,
33,
34,
35,
36,
37,
38,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
50,
51,
52,
53,
54,
55,
56,
57,
58,
59,
60
      ],
      "external_input_tensor_ids": [
"t00002472",
"t00002473",
"t00002474",
"t00002481",
"t00002486",
"t00002487",
"t00001393",
"t00001367",
"t00002511",
"t00000057"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00002472",
"t00002475",
"t00002473",
"t00002477",
"t00002474",
"t00002479",
"t00002481",
"t00002482",
"t00002483",
"t00002486",
"t00002488",
"t00002487",
"t00002490",
"t00002476",
"t00002489",
"t00002494",
"t00002495",
"t00002493",
"t00002496",
"t00002491",
"t00002492",
"t00002497",
"t00002478",
"t00002501",
"t00002502",
"t00002500",
"t00002503",
"t00002499",
"t00002504",
"t00001393",
"t00002505",
"t00001367",
"t00002480",
"t00002506",
"t00002498",
"t00002508",
"t00002509",
"t00002510",
"t00002511",
"t00002512",
"t00002513",
"t00002514",
"t00002507",
"t00002515",
"t00002516",
"t00000057",
"t00002518"
      ],
      "last_event_op_index": 60,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00002484",
"t00002517"
      ],
      "module_path": "model.layers.18.self_attn",
      "module_relative_path": "self_attn",
      "module_type": "VisiPrunerLlamaAttention",
      "op_count": 41,
      "op_counts": {
"add.Tensor": 4,
"cat.default": 4,
"div.Tensor": 1,
"dropout.default": 1,
"gt.Scalar": 1,
"index.Tensor": 2,
"is_nonzero.default": 1,
"matmul.default": 2,
"mul.Tensor": 4,
"neg.default": 2,
"reshape.default": 1,
"select.int": 2,
"slice.Tensor": 4,
"softmax.int": 1,
"to.dtype": 1,
"transpose.int": 5,
"unsqueeze.default": 2,
"view.default": 3
      },
      "output_tensor_ids": [
"t00002475",
"t00002476",
"t00002477",
"t00002478",
"t00002479",
"t00002480",
"t00002482",
"t00002483",
"t00002484",
"t00002488",
"t00002489",
"t00002490",
"t00002491",
"t00002492",
"t00002493",
"t00002494",
"t00002495",
"t00002496",
"t00002497",
"t00002498",
"t00002499",
"t00002500",
"t00002501",
"t00002502",
"t00002503",
"t00002504",
"t00002505",
"t00002506",
"t00002507",
"t00002508",
"t00002509",
"t00002510",
"t00002512",
"t00002513",
"t00002514",
"t00002515",
"t00002516",
"t00002517",
"t00002518"
      ]
    },
    {
      "event_op_indices": [
21,
22,
23,
24,
25,
26,
27,
28
      ],
      "external_input_tensor_ids": [
"t00002484",
"t00001372",
"t00001374"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00002484",
"t00002485",
"t00001372",
"t00002486",
"t00001374",
"t00002487"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.18.self_attn.rotary_emb",
      "module_relative_path": "self_attn.rotary_emb",
      "module_type": "LlamaRotaryEmbedding",
      "op_count": 8,
      "op_counts": {
"gt.Scalar": 1,
"is_nonzero.default": 1,
"item.default": 2,
"slice.Tensor": 2,
"to.dtype": 2
      },
      "output_tensor_ids": [
"t00002485",
"t00002486",
"t00002487"
      ]
    },
    {
      "event_op_indices": [
61
      ],
      "external_input_tensor_ids": [
"t00002517",
"t00001432"
      ],
      "first_event_op_index": 61,
      "input_tensor_ids": [
"t00002517",
"t00001432"
      ],
      "last_event_op_index": 61,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002519"
      ],
      "module_path": "model.layers.18.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002519"
      ]
    },
    {
      "event_op_indices": [
62,
76
      ],
      "external_input_tensor_ids": [
"t00002463",
"t00002519",
"t00002533"
      ],
      "first_event_op_index": 62,
      "input_tensor_ids": [
"t00002463",
"t00002519",
"t00002520",
"t00002533"
      ],
      "last_event_op_index": 76,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00002534"
      ],
      "module_path": "model.layers.18",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00002520",
"t00002534"
      ]
    },
    {
      "event_op_indices": [
63,
64,
65,
66,
67,
68,
69,
70
      ],
      "external_input_tensor_ids": [
"t00002520",
"t00001442"
      ],
      "first_event_op_index": 63,
      "input_tensor_ids": [
"t00002520",
"t00002521",
"t00002522",
"t00002523",
"t00002524",
"t00002525",
"t00002526",
"t00001442",
"t00002527"
      ],
      "last_event_op_index": 70,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00002528"
      ],
      "module_path": "model.layers.18.post_attention_layernorm",
      "module_relative_path": "post_attention_layernorm",
      "module_type": "LlamaRMSNorm",
      "op_count": 8,
      "op_counts": {
"add.Tensor": 1,
"mean.dim": 1,
"mul.Tensor": 2,
"pow.Tensor_Scalar": 1,
"rsqrt.default": 1,
"to.dtype": 2
      },
      "output_tensor_ids": [
"t00002521",
"t00002522",
"t00002523",
"t00002524",
"t00002525",
"t00002526",
"t00002527",
"t00002528"
      ]
    },
    {
      "event_op_indices": [
71
      ],
      "external_input_tensor_ids": [
"t00002528",
"t00001444"
      ],
      "first_event_op_index": 71,
      "input_tensor_ids": [
"t00002528",
"t00001444"
      ],
      "last_event_op_index": 71,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002529"
      ],
      "module_path": "model.layers.18.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002529"
      ]
    },
    {
      "event_op_indices": [
72
      ],
      "external_input_tensor_ids": [
"t00002529"
      ],
      "first_event_op_index": 72,
      "input_tensor_ids": [
"t00002529"
      ],
      "last_event_op_index": 72,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00002530"
      ],
      "module_path": "model.layers.18.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00002530"
      ]
    },
    {
      "event_op_indices": [
73
      ],
      "external_input_tensor_ids": [
"t00002528",
"t00001447"
      ],
      "first_event_op_index": 73,
      "input_tensor_ids": [
"t00002528",
"t00001447"
      ],
      "last_event_op_index": 73,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002531"
      ],
      "module_path": "model.layers.18.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002531"
      ]
    },
    {
      "event_op_indices": [
74
      ],
      "external_input_tensor_ids": [
"t00002530",
"t00002531"
      ],
      "first_event_op_index": 74,
      "input_tensor_ids": [
"t00002530",
"t00002531"
      ],
      "last_event_op_index": 74,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00002532"
      ],
      "module_path": "model.layers.18.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00002532"
      ]
    },
    {
      "event_op_indices": [
75
      ],
      "external_input_tensor_ids": [
"t00002532",
"t00001450"
      ],
      "first_event_op_index": 75,
      "input_tensor_ids": [
"t00002532",
"t00001450"
      ],
      "last_event_op_index": 75,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002533"
      ],
      "module_path": "model.layers.18.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002533"
      ]
    }
  ],
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 4,
    "div.Tensor": 1,
    "dropout.default": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 2,
    "item.default": 2,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 9,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 2,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 2,
    "view.default": 3
  },
  "original_dimensions": {
    "ffn": 11008,
    "head_dim": 128,
    "heads": 32,
    "hidden": 4096,
    "kv_len": 625,
    "seq": 1,
    "tail_start": null,
    "visual_end": null,
    "visual_start": null
  },
  "past_len": 624,
  "phase": "decode",
  "priority": "P3",
  "q_len": 1,
  "row_count": 76,
  "small_config": {
    "ffn": 64,
    "head_dim": 8,
    "heads": 4,
    "hidden": 32,
    "kv_seq": 16,
    "q_seq": 1,
    "seq": 16,
    "tail_start": 13,
    "visual_end": 13,
    "visual_start": 3
  },
  "tensor_dataflow": {
    "edge_count": 85,
    "external_input_tensor_ids": [
      "t00002463",
      "t00001354",
      "t00001356",
      "t00001358",
      "t00001360",
      "t00002481",
      "t00001372",
      "t00001374",
      "t00001393",
      "t00001367",
      "t00002511",
      "t00000057",
      "t00001432",
      "t00001442",
      "t00001444",
      "t00001447",
      "t00001450"
    ],
    "final_output_tensor_ids": [
      "t00002534"
    ],
    "op_count": 76
  },
  "token_state": "full_cache",
  "visipruner_role": "decode_prune_effect"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [1, 4096] -> normalized [1, 4096]
# - qkv_projection: Q/K/V projection: [1, 4096] -> [32, 1, 128]
# - rope: see dispatch evidence for exact tensor roles
# - kv_cache_concat: decode cache concat: current K/V plus past cache -> [32, 625, 128]
# - attention: attention scores: [32, 1, 625]
# - attention_output: attention output: [32, 1, 128] -> [1, 4096]
# - mlp: MLP: [1, 4096] -> [1, 4096]


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> dict[str, torch.Tensor]:
    x_float = hidden_states.to(torch.float32)
    squared = x_float.pow(2)
    variance = squared.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    output = (x_float * inv_rms).to(hidden_states.dtype) * weight
    return {
        "squared": squared,
        "variance": variance,
        "inv_rms": inv_rms,
        "output": output,
    }


def split_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return x.view(x.shape[0], heads, head_dim).transpose(0, 1)


def qkv_projection(x_norm: torch.Tensor, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    heads = int(ORIGINAL_DIMS["heads"])
    head_dim = int(ORIGINAL_DIMS["head_dim"])
    q_linear = F.linear(x_norm, weights["q_weight"])
    k_linear = F.linear(x_norm, weights["k_weight"])
    v_linear = F.linear(x_norm, weights["v_weight"])
    return {
        "q_linear": q_linear,
        "k_linear": k_linear,
        "v_linear": v_linear,
        "q_heads": split_heads(q_linear, heads, head_dim),
        "k_heads_current": split_heads(k_linear, heads, head_dim),
        "v_heads_current": split_heads(v_linear, heads, head_dim),
    }


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(heads_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    cos_for_pos = cos.index_select(0, position_ids).unsqueeze(0)
    sin_for_pos = sin.index_select(0, position_ids).unsqueeze(0)
    return (heads_tensor * cos_for_pos) + (rotate_half(heads_tensor) * sin_for_pos)


def kv_cache_concat(
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    past_k: torch.Tensor | None = None,
    past_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "kv_cache_concat" not in EXPECTED_STAGES:
        return k_current, v_current
    if past_k is None or past_v is None:
        raise ValueError("dispatch expects kv_cache_concat; provide past_k and past_v")
    return torch.cat((past_k, k_current), dim=-2), torch.cat((past_v, v_current), dim=-2)


def attention(q_rope: torch.Tensor, k_rope: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    raw_scores = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    scaled_scores = raw_scores / (q_rope.shape[-1] ** 0.5)
    masked_scores = scaled_scores + attention_mask
    attn = torch.softmax(masked_scores, dim=-1)
    return {
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "masked_scores": masked_scores,
        "attn": attn,
    }


def visual_adjust(attn: torch.Tensor, visual_start: int, visual_end: int, tail_start: int) -> dict[str, torch.Tensor]:
    kind = DISPATCH_FEATURES.get("visual_adjust_kind")
    adjusted = attn.clone()
    if kind is None:
        return {"adjusted_attn": adjusted}
    adjusted[..., visual_start:, visual_start:visual_end] = 0
    if kind == "fold_tail_visual_mass_and_clear_region":
        tail_visual_sum = attn[..., tail_start:, visual_start:visual_end].sum(dim=-1)
        adjusted[..., tail_start:, visual_start] = tail_visual_sum
        return {"adjusted_attn": adjusted, "tail_visual_sum": tail_visual_sum}
    return {"adjusted_attn": adjusted}


def visipruner_similarity_check(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, torch.Tensor]:
    similarity = F.cosine_similarity(reference, candidate, dim=-1)
    threshold = torch.tensor(0.9, dtype=similarity.dtype, device=similarity.device)
    return {"similarity": similarity, "any_close": torch.any(similarity > threshold)}


def attention_output(
    attn: torch.Tensor,
    v_heads: torch.Tensor,
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    context_heads = torch.matmul(attn, v_heads)
    context = context_heads.transpose(0, 1).contiguous().reshape(hidden_states.shape[0], -1)
    attn_out = F.linear(context, weights["o_weight"])
    after_attn = hidden_states + attn_out
    return {"context": context, "attn_out": attn_out, "after_attn": after_attn}


def mlp(after_attn: torch.Tensor, weights: dict[str, torch.Tensor], post_norm_weight: torch.Tensor) -> dict[str, torch.Tensor]:
    post_norm = rms_norm(after_attn, post_norm_weight)
    gate = F.linear(post_norm["output"], weights["gate_weight"])
    up = F.linear(post_norm["output"], weights["up_weight"])
    gated = F.silu(gate) * up
    ffn_out = F.linear(gated, weights["down_weight"])
    output = after_attn + ffn_out
    return {"post_norm": post_norm["output"], "gated": gated, "ffn_out": ffn_out, "output": output}


def expected_stage_evidence() -> dict[str, object]:
    return {stage: CORE_EVIDENCE.get(stage, {}) for stage in EXPECTED_STAGES}
