#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer28, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer28'
ORIGINAL_DIMS = json.loads(r"""{
  "ffn": 11008,
  "head_dim": 128,
  "heads": 32,
  "hidden": 4096,
  "kv_len": 48,
  "q_len": 48
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
  "kv_len": 48,
  "op_counts": {
    "add.Tensor": 10,
    "cat.default": 2,
    "contiguous.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "eq.Scalar": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 2,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 10,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 4,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "sub.Tensor": 1,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 2,
    "view.default": 3
  },
  "phase": "prefill",
  "prune_probe_kind": null,
  "q_len": 48,
  "role": "boundary_after_prune",
  "token_state": "deep_removed",
  "visual_adjust_kind": null
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00002390"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002391"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002392"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002393"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002394"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002395"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00002409",
  "t00002414"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002415"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00002416",
  "t00002421"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002422"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00002422"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 48], dtype=float16",
"output_tensor_ids": [
  "t00002423"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00002415",
  "t00002423"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 48, 48], dtype=float16",
"output_tensor_ids": [
  "t00002424"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00002424"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 48, 48], dtype=float16",
"output_tensor_ids": [
  "t00002425"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00002425",
  "t00002426"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 48, 48], dtype=float16",
"output_tensor_ids": [
  "t00002427"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00002427"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 48, 48], dtype=float32",
"output_tensor_ids": [
  "t00002428"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00002429"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 48, 48], dtype=float16",
"output_tensor_ids": [
  "t00002429"
]
      },
      {
"event_op_index": 54,
"input_tensor_ids": [
  "t00002429",
  "t00002395"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002430"
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
  "t00002429",
  "t00002395"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002430"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00002431"
],
"op_name": "contiguous.default",
"output": "shape=[1, 48, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002432"
]
      },
      {
"event_op_index": 57,
"input_tensor_ids": [
  "t00002432"
],
"op_name": "reshape.default",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002433"
]
      },
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00002433",
  "t00002442"
],
"op_name": "linear.default",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002443"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00002374",
  "t00002443"
],
"op_name": "add.Tensor",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002444"
]
      },
      {
"event_op_index": 82,
"input_tensor_ids": [
  "t00002459",
  "t00002460"
],
"op_name": "linear.default",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002461"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00002444",
  "t00002461"
],
"op_name": "add.Tensor",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002462"
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
  "t00002374"
],
"op_name": "to.dtype",
"output": "shape=[1, 48, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002375"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00002375"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 48, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002376"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00002376"
],
"op_name": "mean.dim",
"output": "shape=[1, 48, 1], dtype=float32",
"output_tensor_ids": [
  "t00002377"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00002377"
],
"op_name": "add.Tensor",
"output": "shape=[1, 48, 1], dtype=float32",
"output_tensor_ids": [
  "t00002378"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00002378"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 48, 1], dtype=float32",
"output_tensor_ids": [
  "t00002379"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00002375",
  "t00002379"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 48, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002380"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00002380"
],
"op_name": "to.dtype",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002381"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00002382",
  "t00002381"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002383"
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
"event_op_index": 62,
"input_tensor_ids": [
  "t00002436"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002437"
]
      },
      {
"event_op_index": 63,
"input_tensor_ids": [
  "t00000057"
],
"op_name": "mul.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002438"
]
      },
      {
"event_op_index": 65,
"input_tensor_ids": [
  "t00002439"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002440"
]
      },
      {
"event_op_index": 68,
"input_tensor_ids": [
  "t00002433",
  "t00002442"
],
"op_name": "linear.default",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002443"
]
      },
      {
"event_op_index": 69,
"input_tensor_ids": [
  "t00002374",
  "t00002443"
],
"op_name": "add.Tensor",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002444"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00002444"
],
"op_name": "to.dtype",
"output": "shape=[1, 48, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002445"
]
      },
      {
"event_op_index": 71,
"input_tensor_ids": [
  "t00002445"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 48, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002446"
]
      },
      {
"event_op_index": 72,
"input_tensor_ids": [
  "t00002446"
],
"op_name": "mean.dim",
"output": "shape=[1, 48, 1], dtype=float32",
"output_tensor_ids": [
  "t00002447"
]
      },
      {
"event_op_index": 73,
"input_tensor_ids": [
  "t00002447"
],
"op_name": "add.Tensor",
"output": "shape=[1, 48, 1], dtype=float32",
"output_tensor_ids": [
  "t00002448"
]
      },
      {
"event_op_index": 74,
"input_tensor_ids": [
  "t00002448"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 48, 1], dtype=float32",
"output_tensor_ids": [
  "t00002449"
]
      },
      {
"event_op_index": 75,
"input_tensor_ids": [
  "t00002445",
  "t00002449"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 48, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002450"
]
      },
      {
"event_op_index": 76,
"input_tensor_ids": [
  "t00002450"
],
"op_name": "to.dtype",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002451"
]
      },
      {
"event_op_index": 77,
"input_tensor_ids": [
  "t00002452",
  "t00002451"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002453"
]
      },
      {
"event_op_index": 78,
"input_tensor_ids": [
  "t00002453",
  "t00002454"
],
"op_name": "linear.default",
"output": "shape=[1, 48, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002455"
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
  "t00002383",
  "t00002384"
],
"op_name": "linear.default",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002385"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00002383",
  "t00002386"
],
"op_name": "linear.default",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002387"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00002383",
  "t00002388"
],
"op_name": "linear.default",
"output": "shape=[1, 48, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002389"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00002385"
],
"op_name": "view.default",
"output": "shape=[1, 48, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002390"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00002390"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002391"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00002387"
],
"op_name": "view.default",
"output": "shape=[1, 48, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002392"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002392"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002393"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00002389"
],
"op_name": "view.default",
"output": "shape=[1, 48, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002394"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002394"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002395"
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
  "t00002398"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002399"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00002401"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00002402"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00002403"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00002404"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00002402",
  "t00002396"
],
"op_name": "index.Tensor",
"output": "shape=[1, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002405"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00002405"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002406"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00002404",
  "t00002396"
],
"op_name": "index.Tensor",
"output": "shape=[1, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002407"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00002407"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002408"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00002391",
  "t00002406"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002409"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00002391"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 48, 64], dtype=float16",
"output_tensor_ids": [
  "t00002410"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00002391"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 48, 64], dtype=float16",
"output_tensor_ids": [
  "t00002411"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00002411"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 48, 64], dtype=float16",
"output_tensor_ids": [
  "t00002412"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00002412",
  "t00002410"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002413"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00002413",
  "t00002408"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002414"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00002409",
  "t00002414"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 48, 128], dtype=float16",
"output_tensor_ids": [
  "t00002415"
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
    "kv_len": 48,
    "op_counts": {
      "add.Tensor": 10,
      "cat.default": 2,
      "contiguous.default": 1,
      "div.Tensor": 1,
      "dropout.default": 1,
      "eq.Scalar": 1,
      "gt.Scalar": 2,
      "index.Tensor": 2,
      "is_nonzero.default": 3,
      "item.default": 2,
      "linear.default": 7,
      "matmul.default": 2,
      "mean.dim": 2,
      "mul.Tensor": 10,
      "neg.default": 2,
      "pow.Tensor_Scalar": 2,
      "reshape.default": 1,
      "rsqrt.default": 2,
      "select.int": 4,
      "silu.default": 1,
      "slice.Tensor": 6,
      "softmax.int": 1,
      "sub.Tensor": 1,
      "to.dtype": 7,
      "transpose.int": 5,
      "unsqueeze.default": 2,
      "view.default": 3
    },
    "phase": "prefill",
    "prune_probe_kind": null,
    "q_len": 48,
    "role": "boundary_after_prune",
    "token_state": "deep_removed",
    "visual_adjust_kind": null
  },
  "dispatch_op_coverage": {
    "covered_op_count": 83,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 83
  },
  "event_id": "input1_layer28",
  "input_id": 1,
  "kv_len": 48,
  "layer_id": 28,
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
"t00002374",
"t00002382"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00002374",
"t00002375",
"t00002376",
"t00002377",
"t00002378",
"t00002379",
"t00002380",
"t00002382",
"t00002381"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00002383"
      ],
      "module_path": "model.layers.28.input_layernorm",
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
"t00002375",
"t00002376",
"t00002377",
"t00002378",
"t00002379",
"t00002380",
"t00002381",
"t00002383"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00002383",
"t00002384"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00002383",
"t00002384"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002385"
      ],
      "module_path": "model.layers.28.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002385"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00002383",
"t00002386"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00002383",
"t00002386"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002387"
      ],
      "module_path": "model.layers.28.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002387"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00002383",
"t00002388"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00002383",
"t00002388"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002389"
      ],
      "module_path": "model.layers.28.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002389"
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
"t00002385",
"t00002387",
"t00002389",
"t00002396",
"t00002402",
"t00002404",
"t00002426",
"t00000057"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00002385",
"t00002390",
"t00002387",
"t00002392",
"t00002389",
"t00002394",
"t00002396",
"t00002397",
"t00002398",
"t00002402",
"t00002405",
"t00002404",
"t00002407",
"t00002391",
"t00002406",
"t00002411",
"t00002412",
"t00002410",
"t00002413",
"t00002408",
"t00002409",
"t00002414",
"t00002393",
"t00002418",
"t00002419",
"t00002417",
"t00002420",
"t00002416",
"t00002421",
"t00002422",
"t00002415",
"t00002423",
"t00002424",
"t00002425",
"t00002426",
"t00002427",
"t00002428",
"t00002429",
"t00002395",
"t00002430",
"t00002431",
"t00002432",
"t00000057",
"t00002434",
"t00002435",
"t00002436",
"t00002437",
"t00002438",
"t00002439",
"t00002440",
"t00002441"
      ],
      "last_event_op_index": 67,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00002399",
"t00002433"
      ],
      "module_path": "model.layers.28.self_attn",
      "module_relative_path": "self_attn",
      "module_type": "VisiPrunerLlamaAttention",
      "op_count": 48,
      "op_counts": {
"add.Tensor": 6,
"cat.default": 2,
"contiguous.default": 1,
"div.Tensor": 1,
"dropout.default": 1,
"eq.Scalar": 1,
"gt.Scalar": 1,
"index.Tensor": 2,
"is_nonzero.default": 2,
"matmul.default": 2,
"mul.Tensor": 5,
"neg.default": 2,
"reshape.default": 1,
"select.int": 4,
"slice.Tensor": 4,
"softmax.int": 1,
"sub.Tensor": 1,
"to.dtype": 1,
"transpose.int": 5,
"unsqueeze.default": 2,
"view.default": 3
      },
      "output_tensor_ids": [
"t00002390",
"t00002391",
"t00002392",
"t00002393",
"t00002394",
"t00002395",
"t00002397",
"t00002398",
"t00002399",
"t00002405",
"t00002406",
"t00002407",
"t00002408",
"t00002409",
"t00002410",
"t00002411",
"t00002412",
"t00002413",
"t00002414",
"t00002415",
"t00002416",
"t00002417",
"t00002418",
"t00002419",
"t00002420",
"t00002421",
"t00002422",
"t00002423",
"t00002424",
"t00002425",
"t00002427",
"t00002428",
"t00002429",
"t00002430",
"t00002431",
"t00002432",
"t00002433",
"t00002434",
"t00002435",
"t00002436",
"t00002437",
"t00002438",
"t00002439",
"t00002440",
"t00002441"
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
"t00002399",
"t00002401",
"t00002403"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00002399",
"t00002400",
"t00002401",
"t00002402",
"t00002403",
"t00002404"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.28.self_attn.rotary_emb",
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
"t00002400",
"t00002402",
"t00002404"
      ]
    },
    {
      "event_op_indices": [
68
      ],
      "external_input_tensor_ids": [
"t00002433",
"t00002442"
      ],
      "first_event_op_index": 68,
      "input_tensor_ids": [
"t00002433",
"t00002442"
      ],
      "last_event_op_index": 68,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002443"
      ],
      "module_path": "model.layers.28.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002443"
      ]
    },
    {
      "event_op_indices": [
69,
83
      ],
      "external_input_tensor_ids": [
"t00002374",
"t00002443",
"t00002461"
      ],
      "first_event_op_index": 69,
      "input_tensor_ids": [
"t00002374",
"t00002443",
"t00002444",
"t00002461"
      ],
      "last_event_op_index": 83,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00002462"
      ],
      "module_path": "model.layers.28",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00002444",
"t00002462"
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
"t00002444",
"t00002452"
      ],
      "first_event_op_index": 70,
      "input_tensor_ids": [
"t00002444",
"t00002445",
"t00002446",
"t00002447",
"t00002448",
"t00002449",
"t00002450",
"t00002452",
"t00002451"
      ],
      "last_event_op_index": 77,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00002453"
      ],
      "module_path": "model.layers.28.post_attention_layernorm",
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
"t00002445",
"t00002446",
"t00002447",
"t00002448",
"t00002449",
"t00002450",
"t00002451",
"t00002453"
      ]
    },
    {
      "event_op_indices": [
78
      ],
      "external_input_tensor_ids": [
"t00002453",
"t00002454"
      ],
      "first_event_op_index": 78,
      "input_tensor_ids": [
"t00002453",
"t00002454"
      ],
      "last_event_op_index": 78,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002455"
      ],
      "module_path": "model.layers.28.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002455"
      ]
    },
    {
      "event_op_indices": [
79
      ],
      "external_input_tensor_ids": [
"t00002455"
      ],
      "first_event_op_index": 79,
      "input_tensor_ids": [
"t00002455"
      ],
      "last_event_op_index": 79,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00002456"
      ],
      "module_path": "model.layers.28.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00002456"
      ]
    },
    {
      "event_op_indices": [
80
      ],
      "external_input_tensor_ids": [
"t00002453",
"t00002457"
      ],
      "first_event_op_index": 80,
      "input_tensor_ids": [
"t00002453",
"t00002457"
      ],
      "last_event_op_index": 80,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002458"
      ],
      "module_path": "model.layers.28.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002458"
      ]
    },
    {
      "event_op_indices": [
81
      ],
      "external_input_tensor_ids": [
"t00002456",
"t00002458"
      ],
      "first_event_op_index": 81,
      "input_tensor_ids": [
"t00002456",
"t00002458"
      ],
      "last_event_op_index": 81,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00002459"
      ],
      "module_path": "model.layers.28.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00002459"
      ]
    },
    {
      "event_op_indices": [
82
      ],
      "external_input_tensor_ids": [
"t00002459",
"t00002460"
      ],
      "first_event_op_index": 82,
      "input_tensor_ids": [
"t00002459",
"t00002460"
      ],
      "last_event_op_index": 82,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002461"
      ],
      "module_path": "model.layers.28.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002461"
      ]
    }
  ],
  "op_counts": {
    "add.Tensor": 10,
    "cat.default": 2,
    "contiguous.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "eq.Scalar": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 2,
    "linear.default": 7,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 10,
    "neg.default": 2,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 4,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "sub.Tensor": 1,
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
    "kv_len": 48,
    "seq": 48,
    "tail_start": null,
    "visual_end": null,
    "visual_start": null
  },
  "past_len": 0,
  "phase": "prefill",
  "priority": "P1",
  "q_len": 48,
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
    "edge_count": 91,
    "external_input_tensor_ids": [
      "t00002374",
      "t00002382",
      "t00002384",
      "t00002386",
      "t00002388",
      "t00002396",
      "t00002401",
      "t00002403",
      "t00002426",
      "t00000057",
      "t00002442",
      "t00002452",
      "t00002454",
      "t00002457",
      "t00002460"
    ],
    "final_output_tensor_ids": [
      "t00002462"
    ],
    "op_count": 83
  },
  "token_state": "deep_removed",
  "visipruner_role": "boundary_after_prune"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [48, 4096] -> normalized [48, 4096]
# - qkv_projection: Q/K/V projection: [48, 4096] -> [32, 48, 128]
# - rope: see dispatch evidence for exact tensor roles
# - attention: attention scores: [32, 48, 48]
# - attention_output: attention output: [32, 48, 128] -> [48, 4096]
# - mlp: MLP: [48, 4096] -> [48, 4096]


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
