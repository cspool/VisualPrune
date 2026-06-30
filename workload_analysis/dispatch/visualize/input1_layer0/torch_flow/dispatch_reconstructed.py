#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer0, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer0'
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
    "copy_.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "fill_.Tensor": 1,
    "gt.Scalar": 3,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 6,
    "lift_fresh.default": 1,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 9,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 3,
    "silu.default": 1,
    "slice.Tensor": 11,
    "softmax.int": 1,
    "sum.dim_IntList": 1,
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
  "visual_adjust_kind": "fold_tail_visual_mass_and_clear_region"
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000017"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000018"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000019"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000020"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000021"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000022"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000036",
  "t00000041"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000042"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00000043",
  "t00000048"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000049"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00000049"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 624], dtype=float16",
"output_tensor_ids": [
  "t00000050"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00000042",
  "t00000050"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000051"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00000051"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000052"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00000052",
  "t00000053"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000054"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00000054"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 624, 624], dtype=float32",
"output_tensor_ids": [
  "t00000055"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00000056"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000056"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00000056",
  "t00000022"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000068"
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
"event_op_index": 70,
"input_tensor_ids": [
  "t00000056",
  "t00000022"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000068"
]
      },
      {
"event_op_index": 72,
"input_tensor_ids": [
  "t00000069"
],
"op_name": "contiguous.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000070"
]
      },
      {
"event_op_index": 73,
"input_tensor_ids": [
  "t00000070"
],
"op_name": "reshape.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000071"
]
      },
      {
"event_op_index": 76,
"input_tensor_ids": [
  "t00000071",
  "t00000073"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000074"
]
      },
      {
"event_op_index": 77,
"input_tensor_ids": [
  "t00000001",
  "t00000074"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000075"
]
      },
      {
"event_op_index": 90,
"input_tensor_ids": [
  "t00000090",
  "t00000091"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000092"
]
      },
      {
"event_op_index": 91,
"input_tensor_ids": [
  "t00000075",
  "t00000092"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000093"
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
  "t00000001"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000002"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00000002"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000003"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00000003"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000004"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00000004"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000005"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00000005"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000006"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00000002",
  "t00000006"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000007"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00000007"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000008"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00000009",
  "t00000008"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000010"
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
"event_op_index": 76,
"input_tensor_ids": [
  "t00000071",
  "t00000073"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000074"
]
      },
      {
"event_op_index": 77,
"input_tensor_ids": [
  "t00000001",
  "t00000074"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000075"
]
      },
      {
"event_op_index": 78,
"input_tensor_ids": [
  "t00000075"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000076"
]
      },
      {
"event_op_index": 79,
"input_tensor_ids": [
  "t00000076"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000077"
]
      },
      {
"event_op_index": 80,
"input_tensor_ids": [
  "t00000077"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000078"
]
      },
      {
"event_op_index": 81,
"input_tensor_ids": [
  "t00000078"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000079"
]
      },
      {
"event_op_index": 82,
"input_tensor_ids": [
  "t00000079"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000080"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00000076",
  "t00000080"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000081"
]
      },
      {
"event_op_index": 84,
"input_tensor_ids": [
  "t00000081"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000082"
]
      },
      {
"event_op_index": 85,
"input_tensor_ids": [
  "t00000083",
  "t00000082"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000084"
]
      },
      {
"event_op_index": 86,
"input_tensor_ids": [
  "t00000084",
  "t00000085"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000086"
]
      },
      {
"event_op_index": 87,
"input_tensor_ids": [
  "t00000086"
],
"op_name": "silu.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000087"
]
      },
      {
"event_op_index": 88,
"input_tensor_ids": [
  "t00000084",
  "t00000088"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000089"
]
      },
      {
"event_op_index": 89,
"input_tensor_ids": [
  "t00000087",
  "t00000089"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000090"
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
  "t00000010",
  "t00000011"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000012"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00000010",
  "t00000013"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000014"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00000010",
  "t00000015"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000016"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00000012"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000017"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000017"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000018"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00000014"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000019"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000019"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000020"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00000016"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000021"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000021"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000022"
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
  "t00000025"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00000026"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00000028"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000029"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00000030"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000031"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00000029",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000032"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00000032"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000033"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00000031",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000034"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00000034"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000035"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00000018",
  "t00000033"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000036"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00000018"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000037"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00000018"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000038"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00000038"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000039"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00000039",
  "t00000037"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000040"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00000040",
  "t00000035"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000041"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000036",
  "t00000041"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000042"
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
"event_op_index": 56,
"input_tensor_ids": [
  "t00000056"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 13, 624], dtype=float16",
"output_tensor_ids": [
  "t00000060"
]
      },
      {
"event_op_index": 58,
"input_tensor_ids": [
  "t00000060"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 13, 576], dtype=float16",
"output_tensor_ids": [
  "t00000061"
]
      },
      {
"event_op_index": 59,
"input_tensor_ids": [
  "t00000061"
],
"op_name": "sum.dim_IntList",
"output": "shape=[1, 32, 13], dtype=float16",
"output_tensor_ids": [
  "t00000062"
]
      },
      {
"event_op_index": 60,
"input_tensor_ids": [
  "t00000063"
],
"op_name": "lift_fresh.default",
"output": "shape=[], dtype=float16",
"output_tensor_ids": [
  "t00000063"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00000056"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 589, 624], dtype=float16",
"output_tensor_ids": [
  "t00000064"
]
      },
      {
"event_op_index": 63,
"input_tensor_ids": [
  "t00000064"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 589, 576], dtype=float16",
"output_tensor_ids": [
  "t00000065"
]
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00000065",
  "t00000063"
],
"op_name": "fill_.Tensor",
"output": "shape=[1, 32, 589, 576], dtype=float16",
"output_tensor_ids": [
  "t00000065"
]
      },
      {
"event_op_index": 66,
"input_tensor_ids": [
  "t00000056"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 13, 624], dtype=float16",
"output_tensor_ids": [
  "t00000066"
]
      },
      {
"event_op_index": 67,
"input_tensor_ids": [
  "t00000066"
],
"op_name": "select.int",
"output": "shape=[1, 32, 13], dtype=float16",
"output_tensor_ids": [
  "t00000067"
]
      },
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00000067",
  "t00000062"
],
"op_name": "copy_.default",
"output": "shape=[1, 32, 13], dtype=float16",
"output_tensor_ids": [
  "t00000067"
]
      }
    ],
    "stage": "visual_adjust",
    "summary": "Visual-adjust evidence kind: fold_tail_visual_mass_and_clear_region."
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
      "copy_.default": 1,
      "div.Tensor": 1,
      "dropout.default": 1,
      "fill_.Tensor": 1,
      "gt.Scalar": 3,
      "index.Tensor": 2,
      "is_nonzero.default": 3,
      "item.default": 6,
      "lift_fresh.default": 1,
      "linear.default": 7,
      "matmul.default": 2,
      "mean.dim": 2,
      "mul.Tensor": 9,
      "neg.default": 2,
      "pow.Tensor_Scalar": 2,
      "reshape.default": 1,
      "rsqrt.default": 2,
      "select.int": 3,
      "silu.default": 1,
      "slice.Tensor": 11,
      "softmax.int": 1,
      "sum.dim_IntList": 1,
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
    "visual_adjust_kind": "fold_tail_visual_mass_and_clear_region"
  },
  "dispatch_op_coverage": {
    "covered_op_count": 91,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 91
  },
  "event_id": "input1_layer0",
  "input_id": 1,
  "kv_len": 624,
  "layer_id": 0,
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
"t00000001",
"t00000009"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00000001",
"t00000002",
"t00000003",
"t00000004",
"t00000005",
"t00000006",
"t00000007",
"t00000009",
"t00000008"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000010"
      ],
      "module_path": "model.layers.0.input_layernorm",
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
"t00000002",
"t00000003",
"t00000004",
"t00000005",
"t00000006",
"t00000007",
"t00000008",
"t00000010"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00000010",
"t00000011"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00000010",
"t00000011"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000012"
      ],
      "module_path": "model.layers.0.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000012"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00000010",
"t00000013"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00000010",
"t00000013"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000014"
      ],
      "module_path": "model.layers.0.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000014"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00000010",
"t00000015"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00000010",
"t00000015"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000016"
      ],
      "module_path": "model.layers.0.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000016"
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
67,
68,
69,
70,
71,
72,
73,
74,
75
      ],
      "external_input_tensor_ids": [
"t00000012",
"t00000014",
"t00000016",
"t00000023",
"t00000029",
"t00000031",
"t00000053",
"t00000057",
"t00000059"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00000012",
"t00000017",
"t00000014",
"t00000019",
"t00000016",
"t00000021",
"t00000023",
"t00000024",
"t00000025",
"t00000029",
"t00000032",
"t00000031",
"t00000034",
"t00000018",
"t00000033",
"t00000038",
"t00000039",
"t00000037",
"t00000040",
"t00000035",
"t00000036",
"t00000041",
"t00000020",
"t00000045",
"t00000046",
"t00000044",
"t00000047",
"t00000043",
"t00000048",
"t00000049",
"t00000042",
"t00000050",
"t00000051",
"t00000052",
"t00000053",
"t00000054",
"t00000055",
"t00000057",
"t00000058",
"t00000059",
"t00000056",
"t00000060",
"t00000061",
"t00000063",
"t00000064",
"t00000065",
"t00000066",
"t00000067",
"t00000062",
"t00000022",
"t00000068",
"t00000069",
"t00000070",
"t00000072"
      ],
      "last_event_op_index": 75,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00000026",
"t00000071"
      ],
      "module_path": "model.layers.0.self_attn",
      "module_relative_path": "self_attn",
      "module_type": "VisiPrunerLlamaAttention",
      "op_count": 56,
      "op_counts": {
"add.Tensor": 4,
"cat.default": 2,
"contiguous.default": 1,
"copy_.default": 1,
"div.Tensor": 1,
"dropout.default": 1,
"fill_.Tensor": 1,
"gt.Scalar": 2,
"index.Tensor": 2,
"is_nonzero.default": 2,
"item.default": 4,
"lift_fresh.default": 1,
"matmul.default": 2,
"mul.Tensor": 4,
"neg.default": 2,
"reshape.default": 1,
"select.int": 3,
"slice.Tensor": 9,
"softmax.int": 1,
"sum.dim_IntList": 1,
"to.dtype": 1,
"transpose.int": 5,
"unsqueeze.default": 2,
"view.default": 3
      },
      "output_tensor_ids": [
"t00000017",
"t00000018",
"t00000019",
"t00000020",
"t00000021",
"t00000022",
"t00000024",
"t00000025",
"t00000026",
"t00000032",
"t00000033",
"t00000034",
"t00000035",
"t00000036",
"t00000037",
"t00000038",
"t00000039",
"t00000040",
"t00000041",
"t00000042",
"t00000043",
"t00000044",
"t00000045",
"t00000046",
"t00000047",
"t00000048",
"t00000049",
"t00000050",
"t00000051",
"t00000052",
"t00000054",
"t00000055",
"t00000056",
"t00000058",
"t00000060",
"t00000061",
"t00000062",
"t00000063",
"t00000064",
"t00000065",
"t00000066",
"t00000067",
"t00000068",
"t00000069",
"t00000070",
"t00000071",
"t00000072"
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
"t00000026",
"t00000028",
"t00000030"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00000026",
"t00000027",
"t00000028",
"t00000029",
"t00000030",
"t00000031"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.0.self_attn.rotary_emb",
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
"t00000027",
"t00000029",
"t00000031"
      ]
    },
    {
      "event_op_indices": [
76
      ],
      "external_input_tensor_ids": [
"t00000071",
"t00000073"
      ],
      "first_event_op_index": 76,
      "input_tensor_ids": [
"t00000071",
"t00000073"
      ],
      "last_event_op_index": 76,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000074"
      ],
      "module_path": "model.layers.0.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000074"
      ]
    },
    {
      "event_op_indices": [
77,
91
      ],
      "external_input_tensor_ids": [
"t00000001",
"t00000074",
"t00000092"
      ],
      "first_event_op_index": 77,
      "input_tensor_ids": [
"t00000001",
"t00000074",
"t00000075",
"t00000092"
      ],
      "last_event_op_index": 91,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00000093"
      ],
      "module_path": "model.layers.0",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00000075",
"t00000093"
      ]
    },
    {
      "event_op_indices": [
78,
79,
80,
81,
82,
83,
84,
85
      ],
      "external_input_tensor_ids": [
"t00000075",
"t00000083"
      ],
      "first_event_op_index": 78,
      "input_tensor_ids": [
"t00000075",
"t00000076",
"t00000077",
"t00000078",
"t00000079",
"t00000080",
"t00000081",
"t00000083",
"t00000082"
      ],
      "last_event_op_index": 85,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000084"
      ],
      "module_path": "model.layers.0.post_attention_layernorm",
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
"t00000076",
"t00000077",
"t00000078",
"t00000079",
"t00000080",
"t00000081",
"t00000082",
"t00000084"
      ]
    },
    {
      "event_op_indices": [
86
      ],
      "external_input_tensor_ids": [
"t00000084",
"t00000085"
      ],
      "first_event_op_index": 86,
      "input_tensor_ids": [
"t00000084",
"t00000085"
      ],
      "last_event_op_index": 86,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000086"
      ],
      "module_path": "model.layers.0.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000086"
      ]
    },
    {
      "event_op_indices": [
87
      ],
      "external_input_tensor_ids": [
"t00000086"
      ],
      "first_event_op_index": 87,
      "input_tensor_ids": [
"t00000086"
      ],
      "last_event_op_index": 87,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00000087"
      ],
      "module_path": "model.layers.0.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00000087"
      ]
    },
    {
      "event_op_indices": [
88
      ],
      "external_input_tensor_ids": [
"t00000084",
"t00000088"
      ],
      "first_event_op_index": 88,
      "input_tensor_ids": [
"t00000084",
"t00000088"
      ],
      "last_event_op_index": 88,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000089"
      ],
      "module_path": "model.layers.0.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000089"
      ]
    },
    {
      "event_op_indices": [
89
      ],
      "external_input_tensor_ids": [
"t00000087",
"t00000089"
      ],
      "first_event_op_index": 89,
      "input_tensor_ids": [
"t00000087",
"t00000089"
      ],
      "last_event_op_index": 89,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00000090"
      ],
      "module_path": "model.layers.0.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00000090"
      ]
    },
    {
      "event_op_indices": [
90
      ],
      "external_input_tensor_ids": [
"t00000090",
"t00000091"
      ],
      "first_event_op_index": 90,
      "input_tensor_ids": [
"t00000090",
"t00000091"
      ],
      "last_event_op_index": 90,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000092"
      ],
      "module_path": "model.layers.0.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000092"
      ]
    }
  ],
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 2,
    "contiguous.default": 1,
    "copy_.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "fill_.Tensor": 1,
    "gt.Scalar": 3,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 6,
    "lift_fresh.default": 1,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 9,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 3,
    "silu.default": 1,
    "slice.Tensor": 11,
    "softmax.int": 1,
    "sum.dim_IntList": 1,
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
  "row_count": 91,
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
    "edge_count": 96,
    "external_input_tensor_ids": [
      "t00000001",
      "t00000009",
      "t00000011",
      "t00000013",
      "t00000015",
      "t00000023",
      "t00000028",
      "t00000030",
      "t00000053",
      "t00000057",
      "t00000059",
      "t00000063",
      "t00000073",
      "t00000083",
      "t00000085",
      "t00000088",
      "t00000091"
    ],
    "final_output_tensor_ids": [
      "t00000093"
    ],
    "op_count": 91
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
