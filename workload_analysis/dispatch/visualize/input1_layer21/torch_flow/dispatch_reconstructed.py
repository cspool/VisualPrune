#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer21, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer21'
ORIGINAL_DIMS = json.loads(r"""{
  "ffn": 11008,
  "head_dim": 128,
  "heads": 32,
  "hidden": 4096,
  "kv_len": 58,
  "q_len": 58
}""")
DISPATCH_FEATURES = json.loads(r"""{
  "expected_stages": [
    "input_rmsnorm",
    "qkv_projection",
    "rope",
    "attention",
    "visipruner_similarity_check",
    "attention_output",
    "mlp"
  ],
  "has_attention": true,
  "has_cache_concat": false,
  "has_mlp": true,
  "has_rope": true,
  "kv_len": 58,
  "op_counts": {
    "add.Tensor": 10,
    "any.default": 1,
    "arange.start": 1,
    "cat.default": 2,
    "contiguous.default": 2,
    "cosine_similarity.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "eq.Scalar": 1,
    "gt.Scalar": 2,
    "index.Tensor": 3,
    "is_nonzero.default": 3,
    "item.default": 3,
    "linear.default": 7,
    "lt.Scalar": 1,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 11,
    "neg.default": 2,
    "permute.default": 1,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 6,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "squeeze.dim": 1,
    "sub.Tensor": 2,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 5,
    "view.default": 4
  },
  "phase": "prefill",
  "prune_probe_kind": "deep_exit_similarity_check",
  "q_len": 58,
  "role": "deep_check",
  "token_state": "middle_pruned",
  "visual_adjust_kind": null
}""")
CORE_EVIDENCE = json.loads(r"""{
  "attention": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00001675"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001676"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00001677"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001678"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00001679"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001680"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00001693",
  "t00001698"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001699"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00001700",
  "t00001705"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001706"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00001706"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 58], dtype=float16",
"output_tensor_ids": [
  "t00001707"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00001699",
  "t00001707"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00001708"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00001708"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00001709"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00001709",
  "t00001505"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00001710"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00001710"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 58, 58], dtype=float32",
"output_tensor_ids": [
  "t00001711"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00001712"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00001712"
]
      },
      {
"event_op_index": 54,
"input_tensor_ids": [
  "t00001712",
  "t00001680"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001713"
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
  "t00001712",
  "t00001680"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001713"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00001714"
],
"op_name": "contiguous.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00001715"
]
      },
      {
"event_op_index": 57,
"input_tensor_ids": [
  "t00001715"
],
"op_name": "reshape.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001716"
]
      },
      {
"event_op_index": 73,
"input_tensor_ids": [
  "t00001729"
],
"op_name": "contiguous.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00001730"
]
      },
      {
"event_op_index": 85,
"input_tensor_ids": [
  "t00001716",
  "t00001741"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001742"
]
      },
      {
"event_op_index": 86,
"input_tensor_ids": [
  "t00001659",
  "t00001742"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001743"
]
      },
      {
"event_op_index": 99,
"input_tensor_ids": [
  "t00001758",
  "t00001759"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001760"
]
      },
      {
"event_op_index": 100,
"input_tensor_ids": [
  "t00001743",
  "t00001760"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001761"
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
  "t00001659"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00001660"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00001660"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00001661"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00001661"
],
"op_name": "mean.dim",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00001662"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00001662"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00001663"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00001663"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00001664"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00001660",
  "t00001664"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00001665"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00001665"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001666"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00001667",
  "t00001666"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001668"
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
"event_op_index": 85,
"input_tensor_ids": [
  "t00001716",
  "t00001741"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001742"
]
      },
      {
"event_op_index": 86,
"input_tensor_ids": [
  "t00001659",
  "t00001742"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001743"
]
      },
      {
"event_op_index": 87,
"input_tensor_ids": [
  "t00001743"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00001744"
]
      },
      {
"event_op_index": 88,
"input_tensor_ids": [
  "t00001744"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00001745"
]
      },
      {
"event_op_index": 89,
"input_tensor_ids": [
  "t00001745"
],
"op_name": "mean.dim",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00001746"
]
      },
      {
"event_op_index": 90,
"input_tensor_ids": [
  "t00001746"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00001747"
]
      },
      {
"event_op_index": 91,
"input_tensor_ids": [
  "t00001747"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00001748"
]
      },
      {
"event_op_index": 92,
"input_tensor_ids": [
  "t00001744",
  "t00001748"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00001749"
]
      },
      {
"event_op_index": 93,
"input_tensor_ids": [
  "t00001749"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001750"
]
      },
      {
"event_op_index": 94,
"input_tensor_ids": [
  "t00001751",
  "t00001750"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001752"
]
      },
      {
"event_op_index": 95,
"input_tensor_ids": [
  "t00001752",
  "t00001753"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00001754"
]
      },
      {
"event_op_index": 96,
"input_tensor_ids": [
  "t00001754"
],
"op_name": "silu.default",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00001755"
]
      },
      {
"event_op_index": 97,
"input_tensor_ids": [
  "t00001752",
  "t00001756"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00001757"
]
      },
      {
"event_op_index": 98,
"input_tensor_ids": [
  "t00001755",
  "t00001757"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00001758"
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
  "t00001668",
  "t00001669"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001670"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00001668",
  "t00001671"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001672"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00001668",
  "t00001673"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001674"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00001670"
],
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00001675"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00001675"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001676"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00001672"
],
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00001677"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00001677"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001678"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00001674"
],
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00001679"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00001679"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001680"
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
  "t00001682"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00001683"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00001685"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00001686"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00001687"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00001688"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00001686",
  "t00001475"
],
"op_name": "index.Tensor",
"output": "shape=[1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001689"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00001689"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001690"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00001688",
  "t00001475"
],
"op_name": "index.Tensor",
"output": "shape=[1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001691"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00001691"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001692"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00001676",
  "t00001690"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001693"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00001676"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 58, 64], dtype=float16",
"output_tensor_ids": [
  "t00001694"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00001676"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 58, 64], dtype=float16",
"output_tensor_ids": [
  "t00001695"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00001695"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 58, 64], dtype=float16",
"output_tensor_ids": [
  "t00001696"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00001696",
  "t00001694"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001697"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00001697",
  "t00001692"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001698"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00001693",
  "t00001698"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00001699"
]
      }
    ],
    "stage": "rope",
    "summary": "RoPE evidence is cos/sin index+unsqueeze, rotate-half slice/neg/cat, then multiply/add."
  },
  "visipruner_similarity_check": {
    "dispatch_supported": true,
    "evidence_ops": [
      {
"event_op_index": 21,
"input_tensor_ids": [
  "t00001683"
],
"op_name": "gt.Scalar",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00001684"
]
      },
      {
"event_op_index": 22,
"input_tensor_ids": [
  "t00001684"
],
"op_name": "is_nonzero.default",
"output": "False",
"output_tensor_ids": []
      },
      {
"event_op_index": 58,
"input_tensor_ids": [
  "t00000057"
],
"op_name": "gt.Scalar",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00001717"
]
      },
      {
"event_op_index": 59,
"input_tensor_ids": [
  "t00001717"
],
"op_name": "is_nonzero.default",
"output": "True",
"output_tensor_ids": []
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00001720",
  "t00001721"
],
"op_name": "sub.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00001722"
]
      },
      {
"event_op_index": 67,
"input_tensor_ids": [
  "t00001724"
],
"op_name": "is_nonzero.default",
"output": "True",
"output_tensor_ids": []
      },
      {
"event_op_index": 75,
"input_tensor_ids": [],
"op_name": "arange.start",
"output": "shape=[10], dtype=int64",
"output_tensor_ids": [
  "t00001732"
]
      },
      {
"event_op_index": 78,
"input_tensor_ids": [
  "t00001734",
  "t00001733"
],
"op_name": "sub.Tensor",
"output": "shape=[1, 10, 4096], dtype=float16",
"output_tensor_ids": [
  "t00001735"
]
      },
      {
"event_op_index": 80,
"input_tensor_ids": [
  "t00001735",
  "t00001736"
],
"op_name": "cosine_similarity.default",
"output": "shape=[1, 10], dtype=float16",
"output_tensor_ids": [
  "t00001737"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00001739"
],
"op_name": "any.default",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00001740"
]
      }
    ],
    "stage": "visipruner_similarity_check",
    "summary": "VisiPrune check evidence kind: deep_exit_similarity_check."
  }
}""")
SUMMARY = json.loads(r"""{
  "dispatch_features": {
    "expected_stages": [
      "input_rmsnorm",
      "qkv_projection",
      "rope",
      "attention",
      "visipruner_similarity_check",
      "attention_output",
      "mlp"
    ],
    "has_attention": true,
    "has_cache_concat": false,
    "has_mlp": true,
    "has_rope": true,
    "kv_len": 58,
    "op_counts": {
      "add.Tensor": 10,
      "any.default": 1,
      "arange.start": 1,
      "cat.default": 2,
      "contiguous.default": 2,
      "cosine_similarity.default": 1,
      "div.Tensor": 1,
      "dropout.default": 1,
      "eq.Scalar": 1,
      "gt.Scalar": 2,
      "index.Tensor": 3,
      "is_nonzero.default": 3,
      "item.default": 3,
      "linear.default": 7,
      "lt.Scalar": 1,
      "matmul.default": 2,
      "mean.dim": 2,
      "mul.Tensor": 11,
      "neg.default": 2,
      "permute.default": 1,
      "pow.Tensor_Scalar": 2,
      "reshape.default": 1,
      "rsqrt.default": 2,
      "select.int": 6,
      "silu.default": 1,
      "slice.Tensor": 6,
      "softmax.int": 1,
      "squeeze.dim": 1,
      "sub.Tensor": 2,
      "to.dtype": 7,
      "transpose.int": 5,
      "unsqueeze.default": 5,
      "view.default": 4
    },
    "phase": "prefill",
    "prune_probe_kind": "deep_exit_similarity_check",
    "q_len": 58,
    "role": "deep_check",
    "token_state": "middle_pruned",
    "visual_adjust_kind": null
  },
  "dispatch_op_coverage": {
    "covered_op_count": 100,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 100
  },
  "event_id": "input1_layer21",
  "input_id": 1,
  "kv_len": 58,
  "layer_id": 21,
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
"t00001659",
"t00001667"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00001659",
"t00001660",
"t00001661",
"t00001662",
"t00001663",
"t00001664",
"t00001665",
"t00001667",
"t00001666"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00001668"
      ],
      "module_path": "model.layers.21.input_layernorm",
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
"t00001660",
"t00001661",
"t00001662",
"t00001663",
"t00001664",
"t00001665",
"t00001666",
"t00001668"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00001668",
"t00001669"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00001668",
"t00001669"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00001670"
      ],
      "module_path": "model.layers.21.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00001670"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00001668",
"t00001671"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00001668",
"t00001671"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00001672"
      ],
      "module_path": "model.layers.21.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00001672"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00001668",
"t00001673"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00001668",
"t00001673"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00001674"
      ],
      "module_path": "model.layers.21.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00001674"
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
75,
76,
77,
78,
79,
80,
81,
82,
83,
84
      ],
      "external_input_tensor_ids": [
"t00001670",
"t00001672",
"t00001674",
"t00001475",
"t00001686",
"t00001688",
"t00001505",
"t00000057"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00001670",
"t00001675",
"t00001672",
"t00001677",
"t00001674",
"t00001679",
"t00001475",
"t00001681",
"t00001682",
"t00001686",
"t00001689",
"t00001688",
"t00001691",
"t00001676",
"t00001690",
"t00001695",
"t00001696",
"t00001694",
"t00001697",
"t00001692",
"t00001693",
"t00001698",
"t00001678",
"t00001702",
"t00001703",
"t00001701",
"t00001704",
"t00001700",
"t00001705",
"t00001706",
"t00001699",
"t00001707",
"t00001708",
"t00001709",
"t00001505",
"t00001710",
"t00001711",
"t00001712",
"t00001680",
"t00001713",
"t00001714",
"t00001715",
"t00000057",
"t00001717",
"t00001718",
"t00001719",
"t00001720",
"t00001721",
"t00001722",
"t00001723",
"t00001724",
"t00001716",
"t00001726",
"t00001727",
"t00001728",
"t00001729",
"t00001730",
"t00001731",
"t00001732",
"t00001725",
"t00001734",
"t00001733",
"t00001735",
"t00001736",
"t00001737",
"t00001738",
"t00001739",
"t00001740"
      ],
      "last_event_op_index": 84,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00001683"
      ],
      "module_path": "model.layers.21.self_attn",
      "module_relative_path": "self_attn",
      "module_type": "VisiPrunerLlamaAttention",
      "op_count": 65,
      "op_counts": {
"add.Tensor": 6,
"any.default": 1,
"arange.start": 1,
"cat.default": 2,
"contiguous.default": 2,
"cosine_similarity.default": 1,
"div.Tensor": 1,
"dropout.default": 1,
"eq.Scalar": 1,
"gt.Scalar": 1,
"index.Tensor": 3,
"is_nonzero.default": 2,
"item.default": 1,
"lt.Scalar": 1,
"matmul.default": 2,
"mul.Tensor": 6,
"neg.default": 2,
"permute.default": 1,
"reshape.default": 1,
"select.int": 6,
"slice.Tensor": 4,
"softmax.int": 1,
"squeeze.dim": 1,
"sub.Tensor": 2,
"to.dtype": 1,
"transpose.int": 5,
"unsqueeze.default": 5,
"view.default": 4
      },
      "output_tensor_ids": [
"t00001675",
"t00001676",
"t00001677",
"t00001678",
"t00001679",
"t00001680",
"t00001681",
"t00001682",
"t00001683",
"t00001689",
"t00001690",
"t00001691",
"t00001692",
"t00001693",
"t00001694",
"t00001695",
"t00001696",
"t00001697",
"t00001698",
"t00001699",
"t00001700",
"t00001701",
"t00001702",
"t00001703",
"t00001704",
"t00001705",
"t00001706",
"t00001707",
"t00001708",
"t00001709",
"t00001710",
"t00001711",
"t00001712",
"t00001713",
"t00001714",
"t00001715",
"t00001716",
"t00001717",
"t00001718",
"t00001719",
"t00001720",
"t00001721",
"t00001722",
"t00001723",
"t00001724",
"t00001725",
"t00001726",
"t00001727",
"t00001728",
"t00001729",
"t00001730",
"t00001731",
"t00001732",
"t00001733",
"t00001734",
"t00001735",
"t00001736",
"t00001737",
"t00001738",
"t00001739",
"t00001740"
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
"t00001683",
"t00001685",
"t00001687"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00001683",
"t00001684",
"t00001685",
"t00001686",
"t00001687",
"t00001688"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.21.self_attn.rotary_emb",
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
"t00001684",
"t00001686",
"t00001688"
      ]
    },
    {
      "event_op_indices": [
85
      ],
      "external_input_tensor_ids": [
"t00001716",
"t00001741"
      ],
      "first_event_op_index": 85,
      "input_tensor_ids": [
"t00001716",
"t00001741"
      ],
      "last_event_op_index": 85,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00001742"
      ],
      "module_path": "model.layers.21.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00001742"
      ]
    },
    {
      "event_op_indices": [
86,
100
      ],
      "external_input_tensor_ids": [
"t00001659",
"t00001742",
"t00001760"
      ],
      "first_event_op_index": 86,
      "input_tensor_ids": [
"t00001659",
"t00001742",
"t00001743",
"t00001760"
      ],
      "last_event_op_index": 100,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00001761"
      ],
      "module_path": "model.layers.21",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00001743",
"t00001761"
      ]
    },
    {
      "event_op_indices": [
87,
88,
89,
90,
91,
92,
93,
94
      ],
      "external_input_tensor_ids": [
"t00001743",
"t00001751"
      ],
      "first_event_op_index": 87,
      "input_tensor_ids": [
"t00001743",
"t00001744",
"t00001745",
"t00001746",
"t00001747",
"t00001748",
"t00001749",
"t00001751",
"t00001750"
      ],
      "last_event_op_index": 94,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00001752"
      ],
      "module_path": "model.layers.21.post_attention_layernorm",
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
"t00001744",
"t00001745",
"t00001746",
"t00001747",
"t00001748",
"t00001749",
"t00001750",
"t00001752"
      ]
    },
    {
      "event_op_indices": [
95
      ],
      "external_input_tensor_ids": [
"t00001752",
"t00001753"
      ],
      "first_event_op_index": 95,
      "input_tensor_ids": [
"t00001752",
"t00001753"
      ],
      "last_event_op_index": 95,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00001754"
      ],
      "module_path": "model.layers.21.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00001754"
      ]
    },
    {
      "event_op_indices": [
96
      ],
      "external_input_tensor_ids": [
"t00001754"
      ],
      "first_event_op_index": 96,
      "input_tensor_ids": [
"t00001754"
      ],
      "last_event_op_index": 96,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00001755"
      ],
      "module_path": "model.layers.21.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00001755"
      ]
    },
    {
      "event_op_indices": [
97
      ],
      "external_input_tensor_ids": [
"t00001752",
"t00001756"
      ],
      "first_event_op_index": 97,
      "input_tensor_ids": [
"t00001752",
"t00001756"
      ],
      "last_event_op_index": 97,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00001757"
      ],
      "module_path": "model.layers.21.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00001757"
      ]
    },
    {
      "event_op_indices": [
98
      ],
      "external_input_tensor_ids": [
"t00001755",
"t00001757"
      ],
      "first_event_op_index": 98,
      "input_tensor_ids": [
"t00001755",
"t00001757"
      ],
      "last_event_op_index": 98,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00001758"
      ],
      "module_path": "model.layers.21.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00001758"
      ]
    },
    {
      "event_op_indices": [
99
      ],
      "external_input_tensor_ids": [
"t00001758",
"t00001759"
      ],
      "first_event_op_index": 99,
      "input_tensor_ids": [
"t00001758",
"t00001759"
      ],
      "last_event_op_index": 99,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00001760"
      ],
      "module_path": "model.layers.21.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00001760"
      ]
    }
  ],
  "op_counts": {
    "add.Tensor": 10,
    "any.default": 1,
    "arange.start": 1,
    "cat.default": 2,
    "contiguous.default": 2,
    "cosine_similarity.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "eq.Scalar": 1,
    "gt.Scalar": 2,
    "index.Tensor": 3,
    "is_nonzero.default": 3,
    "item.default": 3,
    "linear.default": 7,
    "lt.Scalar": 1,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 11,
    "neg.default": 2,
    "permute.default": 1,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 6,
    "silu.default": 1,
    "slice.Tensor": 6,
    "softmax.int": 1,
    "squeeze.dim": 1,
    "sub.Tensor": 2,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 5,
    "view.default": 4
  },
  "original_dimensions": {
    "ffn": 11008,
    "head_dim": 128,
    "heads": 32,
    "hidden": 4096,
    "kv_len": 58,
    "seq": 58,
    "tail_start": null,
    "visual_end": null,
    "visual_start": null
  },
  "past_len": 0,
  "phase": "prefill",
  "priority": "P0",
  "q_len": 58,
  "row_count": 100,
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
    "edge_count": 111,
    "external_input_tensor_ids": [
      "t00001659",
      "t00001667",
      "t00001669",
      "t00001671",
      "t00001673",
      "t00001475",
      "t00001685",
      "t00001687",
      "t00001505",
      "t00000057",
      "t00001741",
      "t00001751",
      "t00001753",
      "t00001756",
      "t00001759"
    ],
    "final_output_tensor_ids": [
      "t00001761"
    ],
    "op_count": 100
  },
  "token_state": "middle_pruned",
  "visipruner_role": "deep_check"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [58, 4096] -> normalized [58, 4096]
# - qkv_projection: Q/K/V projection: [58, 4096] -> [32, 58, 128]
# - rope: see dispatch evidence for exact tensor roles
# - attention: attention scores: [32, 58, 58]
# - visipruner_similarity_check: see dispatch evidence for exact tensor roles
# - attention_output: attention output: [32, 58, 128] -> [58, 4096]
# - mlp: MLP: [58, 4096] -> [58, 4096]


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
