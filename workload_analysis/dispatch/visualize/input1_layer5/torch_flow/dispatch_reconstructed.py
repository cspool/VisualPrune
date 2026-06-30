#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer5, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer5'
ORIGINAL_DIMS = json.loads(r"""{
  "ffn": 11008,
  "head_dim": 128,
  "heads": 32,
  "hidden": 4096,
  "kv_len": 624,
  "q_len": 624
}""")
DISPATCH_FEATURES = json.loads(r"""{
  "expected_stages": [
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "attention",
    "visual_adjust",
    "attention_output",
    "mlp"
  ],
  "has_attention": true,
  "has_cache_concat": false,
  "has_mlp": true,
  "has_rope": true,
  "kv_len": 624,
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 2,
    "contiguous.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "fill_.Tensor": 1,
    "gt.Scalar": 3,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 4,
    "lift_fresh.default": 1,
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
    "slice.Tensor": 8,
    "softmax.int": 1,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 2,
    "view.default": 3
  },
  "phase": "prefill",
  "prune_probe_kind": null,
  "q_len": 624,
  "role": "shallow_or_boundary",
  "token_state": "full_visual",
  "visual_adjust_kind": "clear_visual_region"
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000110"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000111"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000112"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000113"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000114"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000115"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000128",
  "t00000133"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000134"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00000135",
  "t00000140"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000141"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00000141"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 624], dtype=float16",
"output_tensor_ids": [
  "t00000142"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00000134",
  "t00000142"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000143"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00000143"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000144"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00000144",
  "t00000053"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000145"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00000145"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 624, 624], dtype=float32",
"output_tensor_ids": [
  "t00000146"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00000147"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000147"
]
      },
      {
"event_op_index": 62,
"input_tensor_ids": [
  "t00000147",
  "t00000115"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000153"
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
"event_op_index": 62,
"input_tensor_ids": [
  "t00000147",
  "t00000115"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000153"
]
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00000154"
],
"op_name": "contiguous.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000155"
]
      },
      {
"event_op_index": 65,
"input_tensor_ids": [
  "t00000155"
],
"op_name": "reshape.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000156"
]
      },
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00000156",
  "t00000158"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000159"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00000094",
  "t00000159"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000160"
]
      },
      {
"event_op_index": 82,
"input_tensor_ids": [
  "t00000175",
  "t00000176"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000177"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00000160",
  "t00000177"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000178"
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
  "t00000094"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000095"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00000095"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000096"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00000096"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000097"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00000097"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000098"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00000098"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000099"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00000095",
  "t00000099"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000100"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00000100"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000101"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00000102",
  "t00000101"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000103"
]
      }
    ],
    "stage": "input_rmsnorm",
    "summary": "RMSNorm evidence is the initial cast, square, mean, eps-add, rsqrt, and weight multiply sequence."
  },
  "mlp": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00000156",
  "t00000158"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000159"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00000094",
  "t00000159"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000160"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00000160"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000161"
]
      },
      {
"event_op_index": 71,
"input_tensor_ids": [
  "t00000161"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000162"
]
      },
      {
"event_op_index": 72,
"input_tensor_ids": [
  "t00000162"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000163"
]
      },
      {
"event_op_index": 73,
"input_tensor_ids": [
  "t00000163"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000164"
]
      },
      {
"event_op_index": 74,
"input_tensor_ids": [
  "t00000164"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000165"
]
      },
      {
"event_op_index": 75,
"input_tensor_ids": [
  "t00000161",
  "t00000165"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000166"
]
      },
      {
"event_op_index": 76,
"input_tensor_ids": [
  "t00000166"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000167"
]
      },
      {
"event_op_index": 77,
"input_tensor_ids": [
  "t00000168",
  "t00000167"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000169"
]
      },
      {
"event_op_index": 78,
"input_tensor_ids": [
  "t00000169",
  "t00000170"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000171"
]
      },
      {
"event_op_index": 79,
"input_tensor_ids": [
  "t00000171"
],
"op_name": "silu.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000172"
]
      },
      {
"event_op_index": 80,
"input_tensor_ids": [
  "t00000169",
  "t00000173"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000174"
]
      },
      {
"event_op_index": 81,
"input_tensor_ids": [
  "t00000172",
  "t00000174"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000175"
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
  "t00000103",
  "t00000104"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000105"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00000103",
  "t00000106"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000107"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00000103",
  "t00000108"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000109"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00000105"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000110"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000110"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000111"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00000107"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000112"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000112"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000113"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00000109"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000114"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000114"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000115"
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
  "t00000117"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00000118"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00000120"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000121"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00000122"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000123"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00000121",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000124"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00000124"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000125"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00000123",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000126"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00000126"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000127"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00000111",
  "t00000125"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000128"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00000111"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000129"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00000111"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000130"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00000130"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000131"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00000131",
  "t00000129"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000132"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00000132",
  "t00000127"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000133"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000128",
  "t00000133"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000134"
]
      }
    ],
    "stage": "rope",
    "summary": "RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add."
  },
  "visual_adjust": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 55,
"input_tensor_ids": [
  "t00000149"
],
"op_name": "lift_fresh.default",
"output": "shape=[], dtype=float16",
"output_tensor_ids": [
  "t00000149"
]
      },
      {
"event_op_index": 57,
"input_tensor_ids": [
  "t00000147"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 13, 624], dtype=float16",
"output_tensor_ids": [
  "t00000151"
]
      },
      {
"event_op_index": 59,
"input_tensor_ids": [
  "t00000151"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 13, 576], dtype=float16",
"output_tensor_ids": [
  "t00000152"
]
      },
      {
"event_op_index": 60,
"input_tensor_ids": [
  "t00000152",
  "t00000149"
],
"op_name": "fill_.Tensor",
"output": "shape=[1, 32, 13, 576], dtype=float16",
"output_tensor_ids": [
  "t00000152"
]
      }
    ],
    "stage": "visual_adjust",
    "summary": "Visual-adjust evidence kind: clear_visual_region."
  }
}""")
SUMMARY = json.loads(r"""{
  "dispatch_features": {
    "expected_stages": [
      "input_rmsnorm",
      "qkv_projection",
      "rope",
      "attention",
      "visual_adjust",
      "attention_output",
      "mlp"
    ],
    "has_attention": true,
    "has_cache_concat": false,
    "has_mlp": true,
    "has_rope": true,
    "kv_len": 624,
    "op_counts": {
      "add.Tensor": 8,
      "cat.default": 2,
      "contiguous.default": 1,
      "div.Tensor": 1,
      "dropout.default": 1,
      "fill_.Tensor": 1,
      "gt.Scalar": 3,
      "index.Tensor": 2,
      "is_nonzero.default": 3,
      "item.default": 4,
      "lift_fresh.default": 1,
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
      "slice.Tensor": 8,
      "softmax.int": 1,
      "to.dtype": 7,
      "transpose.int": 5,
      "unsqueeze.default": 2,
      "view.default": 3
    },
    "phase": "prefill",
    "prune_probe_kind": null,
    "q_len": 624,
    "role": "shallow_or_boundary",
    "token_state": "full_visual",
    "visual_adjust_kind": "clear_visual_region"
  },
  "dispatch_op_coverage": {
    "covered_op_count": 83,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 83
  },
  "event_id": "input1_layer5",
  "input_id": 1,
  "kv_len": 624,
  "layer_id": 5,
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
"t00000094",
"t00000102"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00000094",
"t00000095",
"t00000096",
"t00000097",
"t00000098",
"t00000099",
"t00000100",
"t00000102",
"t00000101"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000103"
      ],
      "module_path": "model.layers.5.input_layernorm",
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
"t00000095",
"t00000096",
"t00000097",
"t00000098",
"t00000099",
"t00000100",
"t00000101",
"t00000103"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00000103",
"t00000104"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00000103",
"t00000104"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000105"
      ],
      "module_path": "model.layers.5.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000105"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00000103",
"t00000106"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00000103",
"t00000106"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000107"
      ],
      "module_path": "model.layers.5.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000107"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00000103",
"t00000108"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00000103",
"t00000108"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000109"
      ],
      "module_path": "model.layers.5.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000109"
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
60,
61,
62,
63,
64,
65,
66,
67
      ],
      "external_input_tensor_ids": [
"t00000105",
"t00000107",
"t00000109",
"t00000023",
"t00000121",
"t00000123",
"t00000053",
"t00000057",
"t00000150"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00000105",
"t00000110",
"t00000107",
"t00000112",
"t00000109",
"t00000114",
"t00000023",
"t00000116",
"t00000117",
"t00000121",
"t00000124",
"t00000123",
"t00000126",
"t00000111",
"t00000125",
"t00000130",
"t00000131",
"t00000129",
"t00000132",
"t00000127",
"t00000128",
"t00000133",
"t00000113",
"t00000137",
"t00000138",
"t00000136",
"t00000139",
"t00000135",
"t00000140",
"t00000141",
"t00000134",
"t00000142",
"t00000143",
"t00000144",
"t00000053",
"t00000145",
"t00000146",
"t00000057",
"t00000148",
"t00000149",
"t00000150",
"t00000147",
"t00000151",
"t00000152",
"t00000115",
"t00000153",
"t00000154",
"t00000155",
"t00000157"
      ],
      "last_event_op_index": 67,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00000118",
"t00000156"
      ],
      "module_path": "model.layers.5.self_attn",
      "module_relative_path": "self_attn",
      "module_type": "VisiPrunerLlamaAttention",
      "op_count": 48,
      "op_counts": {
"add.Tensor": 4,
"cat.default": 2,
"contiguous.default": 1,
"div.Tensor": 1,
"dropout.default": 1,
"fill_.Tensor": 1,
"gt.Scalar": 2,
"index.Tensor": 2,
"is_nonzero.default": 2,
"item.default": 2,
"lift_fresh.default": 1,
"matmul.default": 2,
"mul.Tensor": 4,
"neg.default": 2,
"reshape.default": 1,
"select.int": 2,
"slice.Tensor": 6,
"softmax.int": 1,
"to.dtype": 1,
"transpose.int": 5,
"unsqueeze.default": 2,
"view.default": 3
      },
      "output_tensor_ids": [
"t00000110",
"t00000111",
"t00000112",
"t00000113",
"t00000114",
"t00000115",
"t00000116",
"t00000117",
"t00000118",
"t00000124",
"t00000125",
"t00000126",
"t00000127",
"t00000128",
"t00000129",
"t00000130",
"t00000131",
"t00000132",
"t00000133",
"t00000134",
"t00000135",
"t00000136",
"t00000137",
"t00000138",
"t00000139",
"t00000140",
"t00000141",
"t00000142",
"t00000143",
"t00000144",
"t00000145",
"t00000146",
"t00000147",
"t00000148",
"t00000149",
"t00000151",
"t00000152",
"t00000153",
"t00000154",
"t00000155",
"t00000156",
"t00000157"
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
"t00000118",
"t00000120",
"t00000122"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00000118",
"t00000119",
"t00000120",
"t00000121",
"t00000122",
"t00000123"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.5.self_attn.rotary_emb",
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
"t00000119",
"t00000121",
"t00000123"
      ]
    },
    {
      "event_op_indices": [
68
      ],
      "external_input_tensor_ids": [
"t00000156",
"t00000158"
      ],
      "first_event_op_index": 68,
      "input_tensor_ids": [
"t00000156",
"t00000158"
      ],
      "last_event_op_index": 68,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000159"
      ],
      "module_path": "model.layers.5.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000159"
      ]
    },
    {
      "event_op_indices": [
69,
83
      ],
      "external_input_tensor_ids": [
"t00000094",
"t00000159",
"t00000177"
      ],
      "first_event_op_index": 69,
      "input_tensor_ids": [
"t00000094",
"t00000159",
"t00000160",
"t00000177"
      ],
      "last_event_op_index": 83,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00000178"
      ],
      "module_path": "model.layers.5",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00000160",
"t00000178"
      ]
    },
    {
      "event_op_indices": [
70,
71,
72,
73,
74,
75,
76,
77
      ],
      "external_input_tensor_ids": [
"t00000160",
"t00000168"
      ],
      "first_event_op_index": 70,
      "input_tensor_ids": [
"t00000160",
"t00000161",
"t00000162",
"t00000163",
"t00000164",
"t00000165",
"t00000166",
"t00000168",
"t00000167"
      ],
      "last_event_op_index": 77,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000169"
      ],
      "module_path": "model.layers.5.post_attention_layernorm",
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
"t00000161",
"t00000162",
"t00000163",
"t00000164",
"t00000165",
"t00000166",
"t00000167",
"t00000169"
      ]
    },
    {
      "event_op_indices": [
78
      ],
      "external_input_tensor_ids": [
"t00000169",
"t00000170"
      ],
      "first_event_op_index": 78,
      "input_tensor_ids": [
"t00000169",
"t00000170"
      ],
      "last_event_op_index": 78,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000171"
      ],
      "module_path": "model.layers.5.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000171"
      ]
    },
    {
      "event_op_indices": [
79
      ],
      "external_input_tensor_ids": [
"t00000171"
      ],
      "first_event_op_index": 79,
      "input_tensor_ids": [
"t00000171"
      ],
      "last_event_op_index": 79,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00000172"
      ],
      "module_path": "model.layers.5.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00000172"
      ]
    },
    {
      "event_op_indices": [
80
      ],
      "external_input_tensor_ids": [
"t00000169",
"t00000173"
      ],
      "first_event_op_index": 80,
      "input_tensor_ids": [
"t00000169",
"t00000173"
      ],
      "last_event_op_index": 80,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000174"
      ],
      "module_path": "model.layers.5.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000174"
      ]
    },
    {
      "event_op_indices": [
81
      ],
      "external_input_tensor_ids": [
"t00000172",
"t00000174"
      ],
      "first_event_op_index": 81,
      "input_tensor_ids": [
"t00000172",
"t00000174"
      ],
      "last_event_op_index": 81,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00000175"
      ],
      "module_path": "model.layers.5.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00000175"
      ]
    },
    {
      "event_op_indices": [
82
      ],
      "external_input_tensor_ids": [
"t00000175",
"t00000176"
      ],
      "first_event_op_index": 82,
      "input_tensor_ids": [
"t00000175",
"t00000176"
      ],
      "last_event_op_index": 82,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000177"
      ],
      "module_path": "model.layers.5.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000177"
      ]
    }
  ],
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 2,
    "contiguous.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "fill_.Tensor": 1,
    "gt.Scalar": 3,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 4,
    "lift_fresh.default": 1,
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
    "slice.Tensor": 8,
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
    "kv_len": 624,
    "seq": 624,
    "tail_start": 611,
    "visual_end": 64,
    "visual_start": 35
  },
  "past_len": 0,
  "phase": "prefill",
  "priority": "P2",
  "q_len": 624,
  "row_count": 83,
  "small_config": {
    "ffn": 64,
    "head_dim": 8,
    "heads": 4,
    "hidden": 32,
    "kv_seq": 16,
    "q_seq": 16,
    "seq": 16,
    "tail_start": 13,
    "visual_end": 13,
    "visual_start": 3
  },
  "tensor_dataflow": {
    "edge_count": 89,
    "external_input_tensor_ids": [
      "t00000094",
      "t00000102",
      "t00000104",
      "t00000106",
      "t00000108",
      "t00000023",
      "t00000120",
      "t00000122",
      "t00000053",
      "t00000057",
      "t00000149",
      "t00000150",
      "t00000158",
      "t00000168",
      "t00000170",
      "t00000173",
      "t00000176"
    ],
    "final_output_tensor_ids": [
      "t00000178"
    ],
    "op_count": 83
  },
  "token_state": "full_visual",
  "visipruner_role": "shallow_or_boundary"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [624, 4096] -> normalized [624, 4096]
# - qkv_projection: Q/K/V projection: [624, 4096] -> [32, 624, 128]
# - rope: see dispatch evidence for exact tensor roles
# - attention: attention scores: [32, 624, 624]
# - visual_adjust: see dispatch evidence for exact tensor roles
# - attention_output: attention output: [32, 624, 128] -> [624, 4096]
# - mlp: MLP: [624, 4096] -> [624, 4096]


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
