#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer6, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer6'
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
  "phase": "prefill",
  "prune_probe_kind": null,
  "q_len": 624,
  "role": "shallow_or_boundary",
  "token_state": "full_visual",
  "visual_adjust_kind": null
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000194"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000195"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000196"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000197"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000198"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000199"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000212",
  "t00000217"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000218"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00000219",
  "t00000224"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000225"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00000225"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 624], dtype=float16",
"output_tensor_ids": [
  "t00000226"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00000218",
  "t00000226"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000227"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00000227"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000228"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00000228",
  "t00000053"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000229"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00000229"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 624, 624], dtype=float32",
"output_tensor_ids": [
  "t00000230"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00000231"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000231"
]
      },
      {
"event_op_index": 54,
"input_tensor_ids": [
  "t00000231",
  "t00000199"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000232"
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
"event_op_index": 54,
"input_tensor_ids": [
  "t00000231",
  "t00000199"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000232"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00000233"
],
"op_name": "contiguous.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000234"
]
      },
      {
"event_op_index": 57,
"input_tensor_ids": [
  "t00000234"
],
"op_name": "reshape.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000235"
]
      },
      {
"event_op_index": 60,
"input_tensor_ids": [
  "t00000235",
  "t00000237"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000238"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00000178",
  "t00000238"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000239"
]
      },
      {
"event_op_index": 74,
"input_tensor_ids": [
  "t00000254",
  "t00000255"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000256"
]
      },
      {
"event_op_index": 75,
"input_tensor_ids": [
  "t00000239",
  "t00000256"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000257"
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
  "t00000178"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000179"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00000179"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000180"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00000180"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000181"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00000181"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000182"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00000182"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000183"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00000179",
  "t00000183"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000184"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00000184"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000185"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00000186",
  "t00000185"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000187"
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
"event_op_index": 52,
"input_tensor_ids": [
  "t00000230"
],
"op_name": "to.dtype",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000231"
]
      },
      {
"event_op_index": 60,
"input_tensor_ids": [
  "t00000235",
  "t00000237"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000238"
]
      },
      {
"event_op_index": 61,
"input_tensor_ids": [
  "t00000178",
  "t00000238"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000239"
]
      },
      {
"event_op_index": 62,
"input_tensor_ids": [
  "t00000239"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000240"
]
      },
      {
"event_op_index": 63,
"input_tensor_ids": [
  "t00000240"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000241"
]
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00000241"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000242"
]
      },
      {
"event_op_index": 65,
"input_tensor_ids": [
  "t00000242"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000243"
]
      },
      {
"event_op_index": 66,
"input_tensor_ids": [
  "t00000243"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000244"
]
      },
      {
"event_op_index": 67,
"input_tensor_ids": [
  "t00000240",
  "t00000244"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000245"
]
      },
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00000245"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000246"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00000247",
  "t00000246"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000248"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00000248",
  "t00000249"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000250"
]
      },
      {
"event_op_index": 71,
"input_tensor_ids": [
  "t00000250"
],
"op_name": "silu.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000251"
]
      },
      {
"event_op_index": 72,
"input_tensor_ids": [
  "t00000248",
  "t00000252"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000253"
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
  "t00000187",
  "t00000188"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000189"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00000187",
  "t00000190"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000191"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00000187",
  "t00000192"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000193"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00000189"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000194"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000194"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000195"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00000191"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000196"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000196"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000197"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00000193"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000198"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000198"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000199"
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
  "t00000201"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00000202"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00000204"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000205"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00000206"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000207"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00000205",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000208"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00000208"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000209"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00000207",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000210"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00000210"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000211"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00000195",
  "t00000209"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000212"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00000195"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000213"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00000195"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000214"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00000214"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000215"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00000215",
  "t00000213"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000216"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00000216",
  "t00000211"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000217"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000212",
  "t00000217"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000218"
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
      "attention",
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
    "phase": "prefill",
    "prune_probe_kind": null,
    "q_len": 624,
    "role": "shallow_or_boundary",
    "token_state": "full_visual",
    "visual_adjust_kind": null
  },
  "dispatch_op_coverage": {
    "covered_op_count": 75,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 75
  },
  "event_id": "input1_layer6",
  "input_id": 1,
  "kv_len": 624,
  "layer_id": 6,
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
"t00000178",
"t00000186"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00000178",
"t00000179",
"t00000180",
"t00000181",
"t00000182",
"t00000183",
"t00000184",
"t00000186",
"t00000185"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000187"
      ],
      "module_path": "model.layers.6.input_layernorm",
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
"t00000179",
"t00000180",
"t00000181",
"t00000182",
"t00000183",
"t00000184",
"t00000185",
"t00000187"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00000187",
"t00000188"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00000187",
"t00000188"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000189"
      ],
      "module_path": "model.layers.6.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000189"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00000187",
"t00000190"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00000187",
"t00000190"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000191"
      ],
      "module_path": "model.layers.6.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000191"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00000187",
"t00000192"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00000187",
"t00000192"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000193"
      ],
      "module_path": "model.layers.6.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000193"
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
59
      ],
      "external_input_tensor_ids": [
"t00000189",
"t00000191",
"t00000193",
"t00000023",
"t00000205",
"t00000207",
"t00000053",
"t00000057"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00000189",
"t00000194",
"t00000191",
"t00000196",
"t00000193",
"t00000198",
"t00000023",
"t00000200",
"t00000201",
"t00000205",
"t00000208",
"t00000207",
"t00000210",
"t00000195",
"t00000209",
"t00000214",
"t00000215",
"t00000213",
"t00000216",
"t00000211",
"t00000212",
"t00000217",
"t00000197",
"t00000221",
"t00000222",
"t00000220",
"t00000223",
"t00000219",
"t00000224",
"t00000225",
"t00000218",
"t00000226",
"t00000227",
"t00000228",
"t00000053",
"t00000229",
"t00000230",
"t00000231",
"t00000199",
"t00000232",
"t00000233",
"t00000234",
"t00000057",
"t00000236"
      ],
      "last_event_op_index": 59,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00000202",
"t00000235"
      ],
      "module_path": "model.layers.6.self_attn",
      "module_relative_path": "self_attn",
      "module_type": "VisiPrunerLlamaAttention",
      "op_count": 40,
      "op_counts": {
"add.Tensor": 4,
"cat.default": 2,
"contiguous.default": 1,
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
"t00000194",
"t00000195",
"t00000196",
"t00000197",
"t00000198",
"t00000199",
"t00000200",
"t00000201",
"t00000202",
"t00000208",
"t00000209",
"t00000210",
"t00000211",
"t00000212",
"t00000213",
"t00000214",
"t00000215",
"t00000216",
"t00000217",
"t00000218",
"t00000219",
"t00000220",
"t00000221",
"t00000222",
"t00000223",
"t00000224",
"t00000225",
"t00000226",
"t00000227",
"t00000228",
"t00000229",
"t00000230",
"t00000231",
"t00000232",
"t00000233",
"t00000234",
"t00000235",
"t00000236"
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
"t00000202",
"t00000204",
"t00000206"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00000202",
"t00000203",
"t00000204",
"t00000205",
"t00000206",
"t00000207"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.6.self_attn.rotary_emb",
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
"t00000203",
"t00000205",
"t00000207"
      ]
    },
    {
      "event_op_indices": [
60
      ],
      "external_input_tensor_ids": [
"t00000235",
"t00000237"
      ],
      "first_event_op_index": 60,
      "input_tensor_ids": [
"t00000235",
"t00000237"
      ],
      "last_event_op_index": 60,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000238"
      ],
      "module_path": "model.layers.6.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000238"
      ]
    },
    {
      "event_op_indices": [
61,
75
      ],
      "external_input_tensor_ids": [
"t00000178",
"t00000238",
"t00000256"
      ],
      "first_event_op_index": 61,
      "input_tensor_ids": [
"t00000178",
"t00000238",
"t00000239",
"t00000256"
      ],
      "last_event_op_index": 75,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00000257"
      ],
      "module_path": "model.layers.6",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00000239",
"t00000257"
      ]
    },
    {
      "event_op_indices": [
62,
63,
64,
65,
66,
67,
68,
69
      ],
      "external_input_tensor_ids": [
"t00000239",
"t00000247"
      ],
      "first_event_op_index": 62,
      "input_tensor_ids": [
"t00000239",
"t00000240",
"t00000241",
"t00000242",
"t00000243",
"t00000244",
"t00000245",
"t00000247",
"t00000246"
      ],
      "last_event_op_index": 69,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000248"
      ],
      "module_path": "model.layers.6.post_attention_layernorm",
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
"t00000240",
"t00000241",
"t00000242",
"t00000243",
"t00000244",
"t00000245",
"t00000246",
"t00000248"
      ]
    },
    {
      "event_op_indices": [
70
      ],
      "external_input_tensor_ids": [
"t00000248",
"t00000249"
      ],
      "first_event_op_index": 70,
      "input_tensor_ids": [
"t00000248",
"t00000249"
      ],
      "last_event_op_index": 70,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000250"
      ],
      "module_path": "model.layers.6.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000250"
      ]
    },
    {
      "event_op_indices": [
71
      ],
      "external_input_tensor_ids": [
"t00000250"
      ],
      "first_event_op_index": 71,
      "input_tensor_ids": [
"t00000250"
      ],
      "last_event_op_index": 71,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00000251"
      ],
      "module_path": "model.layers.6.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00000251"
      ]
    },
    {
      "event_op_indices": [
72
      ],
      "external_input_tensor_ids": [
"t00000248",
"t00000252"
      ],
      "first_event_op_index": 72,
      "input_tensor_ids": [
"t00000248",
"t00000252"
      ],
      "last_event_op_index": 72,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000253"
      ],
      "module_path": "model.layers.6.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000253"
      ]
    },
    {
      "event_op_indices": [
73
      ],
      "external_input_tensor_ids": [
"t00000251",
"t00000253"
      ],
      "first_event_op_index": 73,
      "input_tensor_ids": [
"t00000251",
"t00000253"
      ],
      "last_event_op_index": 73,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00000254"
      ],
      "module_path": "model.layers.6.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00000254"
      ]
    },
    {
      "event_op_indices": [
74
      ],
      "external_input_tensor_ids": [
"t00000254",
"t00000255"
      ],
      "first_event_op_index": 74,
      "input_tensor_ids": [
"t00000254",
"t00000255"
      ],
      "last_event_op_index": 74,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000256"
      ],
      "module_path": "model.layers.6.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000256"
      ]
    }
  ],
  "op_counts": {
    "add.Tensor": 8,
    "cat.default": 2,
    "contiguous.default": 1,
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
    "kv_len": 624,
    "seq": 624,
    "tail_start": null,
    "visual_end": null,
    "visual_start": null
  },
  "past_len": 0,
  "phase": "prefill",
  "priority": "P2",
  "q_len": 624,
  "row_count": 75,
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
    "edge_count": 84,
    "external_input_tensor_ids": [
      "t00000178",
      "t00000186",
      "t00000188",
      "t00000190",
      "t00000192",
      "t00000023",
      "t00000204",
      "t00000206",
      "t00000053",
      "t00000057",
      "t00000237",
      "t00000247",
      "t00000249",
      "t00000252",
      "t00000255"
    ],
    "final_output_tensor_ids": [
      "t00000257"
    ],
    "op_count": 75
  },
  "token_state": "full_visual",
  "visipruner_role": "shallow_or_boundary"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [624, 4096] -> normalized [624, 4096]
# - qkv_projection: Q/K/V projection: [624, 4096] -> [32, 624, 128]
# - rope: see dispatch evidence for exact tensor roles
# - attention: attention scores: [32, 624, 624]
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
