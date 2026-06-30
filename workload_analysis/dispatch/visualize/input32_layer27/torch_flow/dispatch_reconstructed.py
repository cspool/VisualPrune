#!/usr/bin/env python3
"""Torch reconstruction scaffold for input32_layer27, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input32_layer27'
ORIGINAL_DIMS = json.loads(r"""{
  "ffn": 11008,
  "head_dim": 128,
  "heads": 32,
  "hidden": 4096,
  "kv_len": 89,
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
  "kv_len": 89,
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
  "token_state": "middle_pruned_cache",
  "visual_adjust_kind": null
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00002988"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002989"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002990"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002991"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002992"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002993"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00003004",
  "t00003009"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003010"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00003011",
  "t00003016"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003017"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00003019"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 89], dtype=float16",
"output_tensor_ids": [
  "t00003022"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00003010",
  "t00003022"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 89], dtype=float16",
"output_tensor_ids": [
  "t00003023"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00003023"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 1, 89], dtype=float16",
"output_tensor_ids": [
  "t00003024"
]
      },
      {
"event_op_index": 52,
"input_tensor_ids": [
  "t00003024",
  "t00003025"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 89], dtype=float16",
"output_tensor_ids": [
  "t00003026"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00003026"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 1, 89], dtype=float32",
"output_tensor_ids": [
  "t00003027"
]
      },
      {
"event_op_index": 55,
"input_tensor_ids": [
  "t00003028"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 1, 89], dtype=float16",
"output_tensor_ids": [
  "t00003028"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00003028",
  "t00003021"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003029"
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
  "t00003028",
  "t00003021"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003029"
]
      },
      {
"event_op_index": 58,
"input_tensor_ids": [
  "t00003030"
],
"op_name": "reshape.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003031"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00003031",
  "t00002353"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003033"
]
      },
      {
"event_op_index": 62,
"input_tensor_ids": [
  "t00002976",
  "t00003033"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003034"
]
      },
      {
"event_op_index": 75,
"input_tensor_ids": [
  "t00003046",
  "t00002371"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003047"
]
      },
      {
"event_op_index": 76,
"input_tensor_ids": [
  "t00003034",
  "t00003047"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003048"
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
  "t00002976"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002977"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00002977"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002978"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00002978"
],
"op_name": "mean.dim",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002979"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00002979"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002980"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00002980"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00002981"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00002977",
  "t00002981"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002982"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00002982"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002983"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00002279",
  "t00002983"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002984"
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
  "t00003018",
  "t00003017"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 89, 128], dtype=float16",
"output_tensor_ids": [
  "t00003019"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00003020",
  "t00002993"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 89, 128], dtype=float16",
"output_tensor_ids": [
  "t00003021"
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
  "t00003027"
],
"op_name": "to.dtype",
"output": "shape=[1, 32, 1, 89], dtype=float16",
"output_tensor_ids": [
  "t00003028"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00003031",
  "t00002353"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003033"
]
      },
      {
"event_op_index": 62,
"input_tensor_ids": [
  "t00002976",
  "t00003033"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003034"
]
      },
      {
"event_op_index": 63,
"input_tensor_ids": [
  "t00003034"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00003035"
]
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00003035"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00003036"
]
      },
      {
"event_op_index": 65,
"input_tensor_ids": [
  "t00003036"
],
"op_name": "mean.dim",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00003037"
]
      },
      {
"event_op_index": 66,
"input_tensor_ids": [
  "t00003037"
],
"op_name": "add.Tensor",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00003038"
]
      },
      {
"event_op_index": 67,
"input_tensor_ids": [
  "t00003038"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 1, 1], dtype=float32",
"output_tensor_ids": [
  "t00003039"
]
      },
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00003035",
  "t00003039"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float32",
"output_tensor_ids": [
  "t00003040"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00003040"
],
"op_name": "to.dtype",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003041"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00002363",
  "t00003041"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00003042"
]
      },
      {
"event_op_index": 71,
"input_tensor_ids": [
  "t00003042",
  "t00002365"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 11008], dtype=float16",
"output_tensor_ids": [
  "t00003043"
]
      },
      {
"event_op_index": 72,
"input_tensor_ids": [
  "t00003043"
],
"op_name": "silu.default",
"output": "shape=[1, 1, 11008], dtype=float16",
"output_tensor_ids": [
  "t00003044"
]
      },
      {
"event_op_index": 73,
"input_tensor_ids": [
  "t00003042",
  "t00002368"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 11008], dtype=float16",
"output_tensor_ids": [
  "t00003045"
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
  "t00002984",
  "t00002281"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002985"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00002984",
  "t00002283"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002986"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00002984",
  "t00002285"
],
"op_name": "linear.default",
"output": "shape=[1, 1, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002987"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00002985"
],
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002988"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00002988"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002989"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00002986"
],
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002990"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002990"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002991"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00002987"
],
"op_name": "view.default",
"output": "shape=[1, 1, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002992"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002992"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00002993"
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
  "t00002995"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002996"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00002297"
],
"op_name": "slice.Tensor",
"output": "shape=[655, 128], dtype=float16",
"output_tensor_ids": [
  "t00002998"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00002299"
],
"op_name": "slice.Tensor",
"output": "shape=[655, 128], dtype=float16",
"output_tensor_ids": [
  "t00002999"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00002998",
  "t00002848"
],
"op_name": "index.Tensor",
"output": "shape=[1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003000"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00003000"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003001"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00002999",
  "t00002848"
],
"op_name": "index.Tensor",
"output": "shape=[1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003002"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00003002"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003003"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00002989",
  "t00003001"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003004"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00002989"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 1, 64], dtype=float16",
"output_tensor_ids": [
  "t00003005"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00002989"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 1, 64], dtype=float16",
"output_tensor_ids": [
  "t00003006"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00003006"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 1, 64], dtype=float16",
"output_tensor_ids": [
  "t00003007"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00003007",
  "t00003005"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003008"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00003008",
  "t00003003"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003009"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00003004",
  "t00003009"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 1, 128], dtype=float16",
"output_tensor_ids": [
  "t00003010"
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
    "kv_len": 89,
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
    "token_state": "middle_pruned_cache",
    "visual_adjust_kind": null
  },
  "dispatch_op_coverage": {
    "covered_op_count": 76,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 76
  },
  "event_id": "input32_layer27",
  "input_id": 32,
  "kv_len": 89,
  "layer_id": 27,
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
"t00002976",
"t00002279"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00002976",
"t00002977",
"t00002978",
"t00002979",
"t00002980",
"t00002981",
"t00002982",
"t00002279",
"t00002983"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00002984"
      ],
      "module_path": "model.layers.27.input_layernorm",
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
"t00002977",
"t00002978",
"t00002979",
"t00002980",
"t00002981",
"t00002982",
"t00002983",
"t00002984"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00002984",
"t00002281"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00002984",
"t00002281"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002985"
      ],
      "module_path": "model.layers.27.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002985"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00002984",
"t00002283"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00002984",
"t00002283"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002986"
      ],
      "module_path": "model.layers.27.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002986"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00002984",
"t00002285"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00002984",
"t00002285"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002987"
      ],
      "module_path": "model.layers.27.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002987"
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
"t00002985",
"t00002986",
"t00002987",
"t00002848",
"t00002998",
"t00002999",
"t00003018",
"t00003020",
"t00003025",
"t00000057"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00002985",
"t00002988",
"t00002986",
"t00002990",
"t00002987",
"t00002992",
"t00002848",
"t00002994",
"t00002995",
"t00002998",
"t00003000",
"t00002999",
"t00003002",
"t00002989",
"t00003001",
"t00003006",
"t00003007",
"t00003005",
"t00003008",
"t00003003",
"t00003004",
"t00003009",
"t00002991",
"t00003013",
"t00003014",
"t00003012",
"t00003015",
"t00003011",
"t00003016",
"t00003018",
"t00003017",
"t00003020",
"t00002993",
"t00003019",
"t00003010",
"t00003022",
"t00003023",
"t00003024",
"t00003025",
"t00003026",
"t00003027",
"t00003028",
"t00003021",
"t00003029",
"t00003030",
"t00000057",
"t00003032"
      ],
      "last_event_op_index": 60,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00002996",
"t00003031"
      ],
      "module_path": "model.layers.27.self_attn",
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
"t00002988",
"t00002989",
"t00002990",
"t00002991",
"t00002992",
"t00002993",
"t00002994",
"t00002995",
"t00002996",
"t00003000",
"t00003001",
"t00003002",
"t00003003",
"t00003004",
"t00003005",
"t00003006",
"t00003007",
"t00003008",
"t00003009",
"t00003010",
"t00003011",
"t00003012",
"t00003013",
"t00003014",
"t00003015",
"t00003016",
"t00003017",
"t00003019",
"t00003021",
"t00003022",
"t00003023",
"t00003024",
"t00003026",
"t00003027",
"t00003028",
"t00003029",
"t00003030",
"t00003031",
"t00003032"
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
"t00002996",
"t00002297",
"t00002299"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00002996",
"t00002997",
"t00002297",
"t00002998",
"t00002299",
"t00002999"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.27.self_attn.rotary_emb",
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
"t00002997",
"t00002998",
"t00002999"
      ]
    },
    {
      "event_op_indices": [
61
      ],
      "external_input_tensor_ids": [
"t00003031",
"t00002353"
      ],
      "first_event_op_index": 61,
      "input_tensor_ids": [
"t00003031",
"t00002353"
      ],
      "last_event_op_index": 61,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00003033"
      ],
      "module_path": "model.layers.27.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00003033"
      ]
    },
    {
      "event_op_indices": [
62,
76
      ],
      "external_input_tensor_ids": [
"t00002976",
"t00003033",
"t00003047"
      ],
      "first_event_op_index": 62,
      "input_tensor_ids": [
"t00002976",
"t00003033",
"t00003034",
"t00003047"
      ],
      "last_event_op_index": 76,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00003048"
      ],
      "module_path": "model.layers.27",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00003034",
"t00003048"
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
"t00003034",
"t00002363"
      ],
      "first_event_op_index": 63,
      "input_tensor_ids": [
"t00003034",
"t00003035",
"t00003036",
"t00003037",
"t00003038",
"t00003039",
"t00003040",
"t00002363",
"t00003041"
      ],
      "last_event_op_index": 70,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00003042"
      ],
      "module_path": "model.layers.27.post_attention_layernorm",
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
"t00003035",
"t00003036",
"t00003037",
"t00003038",
"t00003039",
"t00003040",
"t00003041",
"t00003042"
      ]
    },
    {
      "event_op_indices": [
71
      ],
      "external_input_tensor_ids": [
"t00003042",
"t00002365"
      ],
      "first_event_op_index": 71,
      "input_tensor_ids": [
"t00003042",
"t00002365"
      ],
      "last_event_op_index": 71,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00003043"
      ],
      "module_path": "model.layers.27.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00003043"
      ]
    },
    {
      "event_op_indices": [
72
      ],
      "external_input_tensor_ids": [
"t00003043"
      ],
      "first_event_op_index": 72,
      "input_tensor_ids": [
"t00003043"
      ],
      "last_event_op_index": 72,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00003044"
      ],
      "module_path": "model.layers.27.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00003044"
      ]
    },
    {
      "event_op_indices": [
73
      ],
      "external_input_tensor_ids": [
"t00003042",
"t00002368"
      ],
      "first_event_op_index": 73,
      "input_tensor_ids": [
"t00003042",
"t00002368"
      ],
      "last_event_op_index": 73,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00003045"
      ],
      "module_path": "model.layers.27.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00003045"
      ]
    },
    {
      "event_op_indices": [
74
      ],
      "external_input_tensor_ids": [
"t00003044",
"t00003045"
      ],
      "first_event_op_index": 74,
      "input_tensor_ids": [
"t00003044",
"t00003045"
      ],
      "last_event_op_index": 74,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00003046"
      ],
      "module_path": "model.layers.27.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00003046"
      ]
    },
    {
      "event_op_indices": [
75
      ],
      "external_input_tensor_ids": [
"t00003046",
"t00002371"
      ],
      "first_event_op_index": 75,
      "input_tensor_ids": [
"t00003046",
"t00002371"
      ],
      "last_event_op_index": 75,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00003047"
      ],
      "module_path": "model.layers.27.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00003047"
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
    "kv_len": 89,
    "seq": 1,
    "tail_start": null,
    "visual_end": null,
    "visual_start": null
  },
  "past_len": 88,
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
      "t00002976",
      "t00002279",
      "t00002281",
      "t00002283",
      "t00002285",
      "t00002848",
      "t00002297",
      "t00002299",
      "t00003018",
      "t00003020",
      "t00003025",
      "t00000057",
      "t00002353",
      "t00002363",
      "t00002365",
      "t00002368",
      "t00002371"
    ],
    "final_output_tensor_ids": [
      "t00003048"
    ],
    "op_count": 76
  },
  "token_state": "middle_pruned_cache",
  "visipruner_role": "decode_prune_effect"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [1, 4096] -> normalized [1, 4096]
# - qkv_projection: Q/K/V projection: [1, 4096] -> [32, 1, 128]
# - rope: see dispatch evidence for exact tensor roles
# - kv_cache_concat: decode cache concat: current K/V plus past cache -> [32, 89, 128]
# - attention: attention scores: [32, 1, 89]
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
