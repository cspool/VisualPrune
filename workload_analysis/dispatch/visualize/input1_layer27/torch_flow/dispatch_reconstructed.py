#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer27, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer27'
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
  "role": "deep_check;boundary_before_prune",
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
  "t00002287"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002288"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002289"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002290"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002291"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002292"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00002305",
  "t00002310"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002311"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00002312",
  "t00002317"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002318"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00002318"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 58], dtype=float16",
"output_tensor_ids": [
  "t00002319"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00002311",
  "t00002319"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00002320"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00002320"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00002321"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00002321",
  "t00001505"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00002322"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00002322"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 58, 58], dtype=float32",
"output_tensor_ids": [
  "t00002323"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00002324"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 58, 58], dtype=float16",
"output_tensor_ids": [
  "t00002324"
]
      },
      {
"event_op_index": 54,
"input_tensor_ids": [
  "t00002324",
  "t00002292"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002325"
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
  "t00002324",
  "t00002292"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002325"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00002326"
],
"op_name": "contiguous.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002327"
]
      },
      {
"event_op_index": 57,
"input_tensor_ids": [
  "t00002327"
],
"op_name": "reshape.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002328"
]
      },
      {
"event_op_index": 73,
"input_tensor_ids": [
  "t00002341"
],
"op_name": "contiguous.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002342"
]
      },
      {
"event_op_index": 85,
"input_tensor_ids": [
  "t00002328",
  "t00002353"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002354"
]
      },
      {
"event_op_index": 86,
"input_tensor_ids": [
  "t00002271",
  "t00002354"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002355"
]
      },
      {
"event_op_index": 99,
"input_tensor_ids": [
  "t00002370",
  "t00002371"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002372"
]
      },
      {
"event_op_index": 100,
"input_tensor_ids": [
  "t00002355",
  "t00002372"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002373"
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
  "t00002271"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002272"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00002272"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002273"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00002273"
],
"op_name": "mean.dim",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00002274"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00002274"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00002275"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00002275"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00002276"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00002272",
  "t00002276"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002277"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00002277"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002278"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00002279",
  "t00002278"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002280"
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
  "t00002328",
  "t00002353"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002354"
]
      },
      {
"event_op_index": 86,
"input_tensor_ids": [
  "t00002271",
  "t00002354"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002355"
]
      },
      {
"event_op_index": 87,
"input_tensor_ids": [
  "t00002355"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002356"
]
      },
      {
"event_op_index": 88,
"input_tensor_ids": [
  "t00002356"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002357"
]
      },
      {
"event_op_index": 89,
"input_tensor_ids": [
  "t00002357"
],
"op_name": "mean.dim",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00002358"
]
      },
      {
"event_op_index": 90,
"input_tensor_ids": [
  "t00002358"
],
"op_name": "add.Tensor",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00002359"
]
      },
      {
"event_op_index": 91,
"input_tensor_ids": [
  "t00002359"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 58, 1], dtype=float32",
"output_tensor_ids": [
  "t00002360"
]
      },
      {
"event_op_index": 92,
"input_tensor_ids": [
  "t00002356",
  "t00002360"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float32",
"output_tensor_ids": [
  "t00002361"
]
      },
      {
"event_op_index": 93,
"input_tensor_ids": [
  "t00002361"
],
"op_name": "to.dtype",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002362"
]
      },
      {
"event_op_index": 94,
"input_tensor_ids": [
  "t00002363",
  "t00002362"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002364"
]
      },
      {
"event_op_index": 95,
"input_tensor_ids": [
  "t00002364",
  "t00002365"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002366"
]
      },
      {
"event_op_index": 96,
"input_tensor_ids": [
  "t00002366"
],
"op_name": "silu.default",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002367"
]
      },
      {
"event_op_index": 97,
"input_tensor_ids": [
  "t00002364",
  "t00002368"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002369"
]
      },
      {
"event_op_index": 98,
"input_tensor_ids": [
  "t00002367",
  "t00002369"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 58, 11008], dtype=float16",
"output_tensor_ids": [
  "t00002370"
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
  "t00002280",
  "t00002281"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002282"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00002280",
  "t00002283"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002284"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00002280",
  "t00002285"
],
"op_name": "linear.default",
"output": "shape=[1, 58, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002286"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00002282"
],
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002287"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00002287"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002288"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00002284"
],
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002289"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00002289"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002290"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00002286"
],
"op_name": "view.default",
"output": "shape=[1, 58, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00002291"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00002291"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002292"
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
  "t00002294"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002295"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00002297"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00002298"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00002299"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00002300"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00002298",
  "t00001475"
],
"op_name": "index.Tensor",
"output": "shape=[1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002301"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00002301"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002302"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00002300",
  "t00001475"
],
"op_name": "index.Tensor",
"output": "shape=[1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002303"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00002303"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002304"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00002288",
  "t00002302"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002305"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00002288"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 58, 64], dtype=float16",
"output_tensor_ids": [
  "t00002306"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00002288"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 58, 64], dtype=float16",
"output_tensor_ids": [
  "t00002307"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00002307"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 58, 64], dtype=float16",
"output_tensor_ids": [
  "t00002308"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00002308",
  "t00002306"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002309"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00002309",
  "t00002304"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002310"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00002305",
  "t00002310"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 58, 128], dtype=float16",
"output_tensor_ids": [
  "t00002311"
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
  "t00002295"
],
"op_name": "gt.Scalar",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00002296"
]
      },
      {
"event_op_index": 22,
"input_tensor_ids": [
  "t00002296"
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
  "t00002329"
]
      },
      {
"event_op_index": 59,
"input_tensor_ids": [
  "t00002329"
],
"op_name": "is_nonzero.default",
"output": "True",
"output_tensor_ids": []
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00002332",
  "t00002333"
],
"op_name": "sub.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00002334"
]
      },
      {
"event_op_index": 67,
"input_tensor_ids": [
  "t00002336"
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
  "t00002344"
]
      },
      {
"event_op_index": 78,
"input_tensor_ids": [
  "t00002346",
  "t00002345"
],
"op_name": "sub.Tensor",
"output": "shape=[1, 10, 4096], dtype=float16",
"output_tensor_ids": [
  "t00002347"
]
      },
      {
"event_op_index": 80,
"input_tensor_ids": [
  "t00002347",
  "t00002348"
],
"op_name": "cosine_similarity.default",
"output": "shape=[1, 10], dtype=float16",
"output_tensor_ids": [
  "t00002349"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00002351"
],
"op_name": "any.default",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00002352"
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
    "role": "deep_check;boundary_before_prune",
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
  "event_id": "input1_layer27",
  "input_id": 1,
  "kv_len": 58,
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
"t00002271",
"t00002279"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00002271",
"t00002272",
"t00002273",
"t00002274",
"t00002275",
"t00002276",
"t00002277",
"t00002279",
"t00002278"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00002280"
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
"t00002272",
"t00002273",
"t00002274",
"t00002275",
"t00002276",
"t00002277",
"t00002278",
"t00002280"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00002280",
"t00002281"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00002280",
"t00002281"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002282"
      ],
      "module_path": "model.layers.27.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002282"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00002280",
"t00002283"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00002280",
"t00002283"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002284"
      ],
      "module_path": "model.layers.27.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002284"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00002280",
"t00002285"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00002280",
"t00002285"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002286"
      ],
      "module_path": "model.layers.27.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002286"
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
"t00002282",
"t00002284",
"t00002286",
"t00001475",
"t00002298",
"t00002300",
"t00001505",
"t00000057"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00002282",
"t00002287",
"t00002284",
"t00002289",
"t00002286",
"t00002291",
"t00001475",
"t00002293",
"t00002294",
"t00002298",
"t00002301",
"t00002300",
"t00002303",
"t00002288",
"t00002302",
"t00002307",
"t00002308",
"t00002306",
"t00002309",
"t00002304",
"t00002305",
"t00002310",
"t00002290",
"t00002314",
"t00002315",
"t00002313",
"t00002316",
"t00002312",
"t00002317",
"t00002318",
"t00002311",
"t00002319",
"t00002320",
"t00002321",
"t00001505",
"t00002322",
"t00002323",
"t00002324",
"t00002292",
"t00002325",
"t00002326",
"t00002327",
"t00000057",
"t00002329",
"t00002330",
"t00002331",
"t00002332",
"t00002333",
"t00002334",
"t00002335",
"t00002336",
"t00002328",
"t00002338",
"t00002339",
"t00002340",
"t00002341",
"t00002342",
"t00002343",
"t00002344",
"t00002337",
"t00002346",
"t00002345",
"t00002347",
"t00002348",
"t00002349",
"t00002350",
"t00002351",
"t00002352"
      ],
      "last_event_op_index": 84,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00002295"
      ],
      "module_path": "model.layers.27.self_attn",
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
"t00002287",
"t00002288",
"t00002289",
"t00002290",
"t00002291",
"t00002292",
"t00002293",
"t00002294",
"t00002295",
"t00002301",
"t00002302",
"t00002303",
"t00002304",
"t00002305",
"t00002306",
"t00002307",
"t00002308",
"t00002309",
"t00002310",
"t00002311",
"t00002312",
"t00002313",
"t00002314",
"t00002315",
"t00002316",
"t00002317",
"t00002318",
"t00002319",
"t00002320",
"t00002321",
"t00002322",
"t00002323",
"t00002324",
"t00002325",
"t00002326",
"t00002327",
"t00002328",
"t00002329",
"t00002330",
"t00002331",
"t00002332",
"t00002333",
"t00002334",
"t00002335",
"t00002336",
"t00002337",
"t00002338",
"t00002339",
"t00002340",
"t00002341",
"t00002342",
"t00002343",
"t00002344",
"t00002345",
"t00002346",
"t00002347",
"t00002348",
"t00002349",
"t00002350",
"t00002351",
"t00002352"
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
"t00002295",
"t00002297",
"t00002299"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00002295",
"t00002296",
"t00002297",
"t00002298",
"t00002299",
"t00002300"
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
"t00002296",
"t00002298",
"t00002300"
      ]
    },
    {
      "event_op_indices": [
85
      ],
      "external_input_tensor_ids": [
"t00002328",
"t00002353"
      ],
      "first_event_op_index": 85,
      "input_tensor_ids": [
"t00002328",
"t00002353"
      ],
      "last_event_op_index": 85,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002354"
      ],
      "module_path": "model.layers.27.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002354"
      ]
    },
    {
      "event_op_indices": [
86,
100
      ],
      "external_input_tensor_ids": [
"t00002271",
"t00002354",
"t00002372"
      ],
      "first_event_op_index": 86,
      "input_tensor_ids": [
"t00002271",
"t00002354",
"t00002355",
"t00002372"
      ],
      "last_event_op_index": 100,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00002373"
      ],
      "module_path": "model.layers.27",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00002355",
"t00002373"
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
"t00002355",
"t00002363"
      ],
      "first_event_op_index": 87,
      "input_tensor_ids": [
"t00002355",
"t00002356",
"t00002357",
"t00002358",
"t00002359",
"t00002360",
"t00002361",
"t00002363",
"t00002362"
      ],
      "last_event_op_index": 94,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00002364"
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
"t00002356",
"t00002357",
"t00002358",
"t00002359",
"t00002360",
"t00002361",
"t00002362",
"t00002364"
      ]
    },
    {
      "event_op_indices": [
95
      ],
      "external_input_tensor_ids": [
"t00002364",
"t00002365"
      ],
      "first_event_op_index": 95,
      "input_tensor_ids": [
"t00002364",
"t00002365"
      ],
      "last_event_op_index": 95,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002366"
      ],
      "module_path": "model.layers.27.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002366"
      ]
    },
    {
      "event_op_indices": [
96
      ],
      "external_input_tensor_ids": [
"t00002366"
      ],
      "first_event_op_index": 96,
      "input_tensor_ids": [
"t00002366"
      ],
      "last_event_op_index": 96,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00002367"
      ],
      "module_path": "model.layers.27.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00002367"
      ]
    },
    {
      "event_op_indices": [
97
      ],
      "external_input_tensor_ids": [
"t00002364",
"t00002368"
      ],
      "first_event_op_index": 97,
      "input_tensor_ids": [
"t00002364",
"t00002368"
      ],
      "last_event_op_index": 97,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002369"
      ],
      "module_path": "model.layers.27.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002369"
      ]
    },
    {
      "event_op_indices": [
98
      ],
      "external_input_tensor_ids": [
"t00002367",
"t00002369"
      ],
      "first_event_op_index": 98,
      "input_tensor_ids": [
"t00002367",
"t00002369"
      ],
      "last_event_op_index": 98,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00002370"
      ],
      "module_path": "model.layers.27.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00002370"
      ]
    },
    {
      "event_op_indices": [
99
      ],
      "external_input_tensor_ids": [
"t00002370",
"t00002371"
      ],
      "first_event_op_index": 99,
      "input_tensor_ids": [
"t00002370",
"t00002371"
      ],
      "last_event_op_index": 99,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00002372"
      ],
      "module_path": "model.layers.27.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00002372"
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
      "t00002271",
      "t00002279",
      "t00002281",
      "t00002283",
      "t00002285",
      "t00001475",
      "t00002297",
      "t00002299",
      "t00001505",
      "t00000057",
      "t00002353",
      "t00002363",
      "t00002365",
      "t00002368",
      "t00002371"
    ],
    "final_output_tensor_ids": [
      "t00002373"
    ],
    "op_count": 100
  },
  "token_state": "middle_pruned",
  "visipruner_role": "deep_check;boundary_before_prune"
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
