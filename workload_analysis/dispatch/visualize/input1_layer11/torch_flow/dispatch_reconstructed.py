#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer11, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer11'
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
    "visipruner_similarity_check",
    "attention_output",
    "mlp"
  ],
  "has_attention": true,
  "has_cache_concat": false,
  "has_mlp": true,
  "has_rope": true,
  "kv_len": 624,
  "op_counts": {
    "add.Tensor": 9,
    "any.default": 1,
    "cat.default": 2,
    "contiguous.default": 2,
    "cosine_similarity.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "eq.Scalar": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 4,
    "linear.default": 7,
    "lt.Scalar": 1,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 10,
    "neg.default": 2,
    "permute.default": 1,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 6,
    "silu.default": 1,
    "slice.Tensor": 7,
    "softmax.int": 1,
    "squeeze.dim": 1,
    "sub.Tensor": 1,
    "to.dtype": 7,
    "transpose.int": 5,
    "unsqueeze.default": 5,
    "view.default": 4
  },
  "phase": "prefill",
  "prune_probe_kind": "middle_probe_similarity_check",
  "q_len": 624,
  "role": "middle_probe",
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
  "t00000669"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000670"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000671"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000672"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000673"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000674"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000687",
  "t00000692"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000693"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00000694",
  "t00000699"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000700"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00000700"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 624], dtype=float16",
"output_tensor_ids": [
  "t00000701"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00000693",
  "t00000701"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000702"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00000702"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000703"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00000703",
  "t00000053"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000704"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00000704"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 624, 624], dtype=float32",
"output_tensor_ids": [
  "t00000705"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00000706"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000706"
]
      },
      {
"event_op_index": 54,
"input_tensor_ids": [
  "t00000706",
  "t00000674"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000707"
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
  "t00000706",
  "t00000674"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000707"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00000708"
],
"op_name": "contiguous.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000709"
]
      },
      {
"event_op_index": 57,
"input_tensor_ids": [
  "t00000709"
],
"op_name": "reshape.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000710"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00000720"
],
"op_name": "contiguous.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000721"
]
      },
      {
"event_op_index": 82,
"input_tensor_ids": [
  "t00000710",
  "t00000732"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000733"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00000653",
  "t00000733"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000734"
]
      },
      {
"event_op_index": 96,
"input_tensor_ids": [
  "t00000749",
  "t00000750"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000751"
]
      },
      {
"event_op_index": 97,
"input_tensor_ids": [
  "t00000734",
  "t00000751"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000752"
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
  "t00000653"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000654"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00000654"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000655"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00000655"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000656"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00000656"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000657"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00000657"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000658"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00000654",
  "t00000658"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000659"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00000659"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000660"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00000661",
  "t00000660"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000662"
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
"event_op_index": 82,
"input_tensor_ids": [
  "t00000710",
  "t00000732"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000733"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00000653",
  "t00000733"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000734"
]
      },
      {
"event_op_index": 84,
"input_tensor_ids": [
  "t00000734"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000735"
]
      },
      {
"event_op_index": 85,
"input_tensor_ids": [
  "t00000735"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000736"
]
      },
      {
"event_op_index": 86,
"input_tensor_ids": [
  "t00000736"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000737"
]
      },
      {
"event_op_index": 87,
"input_tensor_ids": [
  "t00000737"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000738"
]
      },
      {
"event_op_index": 88,
"input_tensor_ids": [
  "t00000738"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000739"
]
      },
      {
"event_op_index": 89,
"input_tensor_ids": [
  "t00000735",
  "t00000739"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000740"
]
      },
      {
"event_op_index": 90,
"input_tensor_ids": [
  "t00000740"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000741"
]
      },
      {
"event_op_index": 91,
"input_tensor_ids": [
  "t00000742",
  "t00000741"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000743"
]
      },
      {
"event_op_index": 92,
"input_tensor_ids": [
  "t00000743",
  "t00000744"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000745"
]
      },
      {
"event_op_index": 93,
"input_tensor_ids": [
  "t00000745"
],
"op_name": "silu.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000746"
]
      },
      {
"event_op_index": 94,
"input_tensor_ids": [
  "t00000743",
  "t00000747"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000748"
]
      },
      {
"event_op_index": 95,
"input_tensor_ids": [
  "t00000746",
  "t00000748"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000749"
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
  "t00000662",
  "t00000663"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000664"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00000662",
  "t00000665"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000666"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00000662",
  "t00000667"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000668"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00000664"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000669"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000669"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000670"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00000666"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000671"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000671"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000672"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00000668"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000673"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000673"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000674"
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
  "t00000676"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00000677"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00000679"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000680"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00000681"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000682"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00000680",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000683"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00000683"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000684"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00000682",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000685"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00000685"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000686"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00000670",
  "t00000684"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000687"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00000670"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000688"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00000670"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000689"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00000689"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000690"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00000690",
  "t00000688"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000691"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00000691",
  "t00000686"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000692"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000687",
  "t00000692"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000693"
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
  "t00000677"
],
"op_name": "gt.Scalar",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00000678"
]
      },
      {
"event_op_index": 22,
"input_tensor_ids": [
  "t00000678"
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
  "t00000711"
]
      },
      {
"event_op_index": 59,
"input_tensor_ids": [
  "t00000711"
],
"op_name": "is_nonzero.default",
"output": "True",
"output_tensor_ids": []
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00000715"
],
"op_name": "is_nonzero.default",
"output": "True",
"output_tensor_ids": []
      },
      {
"event_op_index": 75,
"input_tensor_ids": [
  "t00000725",
  "t00000724"
],
"op_name": "sub.Tensor",
"output": "shape=[1, 576, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000726"
]
      },
      {
"event_op_index": 77,
"input_tensor_ids": [
  "t00000726",
  "t00000727"
],
"op_name": "cosine_similarity.default",
"output": "shape=[1, 576], dtype=float16",
"output_tensor_ids": [
  "t00000728"
]
      },
      {
"event_op_index": 80,
"input_tensor_ids": [
  "t00000730"
],
"op_name": "any.default",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00000731"
]
      }
    ],
    "stage": "visipruner_similarity_check",
    "summary": "VisiPrune check evidence kind: middle_probe_similarity_check."
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
    "kv_len": 624,
    "op_counts": {
      "add.Tensor": 9,
      "any.default": 1,
      "cat.default": 2,
      "contiguous.default": 2,
      "cosine_similarity.default": 1,
      "div.Tensor": 1,
      "dropout.default": 1,
      "eq.Scalar": 1,
      "gt.Scalar": 2,
      "index.Tensor": 2,
      "is_nonzero.default": 3,
      "item.default": 4,
      "linear.default": 7,
      "lt.Scalar": 1,
      "matmul.default": 2,
      "mean.dim": 2,
      "mul.Tensor": 10,
      "neg.default": 2,
      "permute.default": 1,
      "pow.Tensor_Scalar": 2,
      "reshape.default": 1,
      "rsqrt.default": 2,
      "select.int": 6,
      "silu.default": 1,
      "slice.Tensor": 7,
      "softmax.int": 1,
      "squeeze.dim": 1,
      "sub.Tensor": 1,
      "to.dtype": 7,
      "transpose.int": 5,
      "unsqueeze.default": 5,
      "view.default": 4
    },
    "phase": "prefill",
    "prune_probe_kind": "middle_probe_similarity_check",
    "q_len": 624,
    "role": "middle_probe",
    "token_state": "full_visual",
    "visual_adjust_kind": null
  },
  "dispatch_op_coverage": {
    "covered_op_count": 97,
    "missing_event_op_indices": [],
    "missing_from_module_split": [],
    "missing_from_tensor_dataflow": [],
    "op_count": 97
  },
  "event_id": "input1_layer11",
  "input_id": 1,
  "kv_len": 624,
  "layer_id": 11,
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
"t00000653",
"t00000661"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00000653",
"t00000654",
"t00000655",
"t00000656",
"t00000657",
"t00000658",
"t00000659",
"t00000661",
"t00000660"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000662"
      ],
      "module_path": "model.layers.11.input_layernorm",
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
"t00000654",
"t00000655",
"t00000656",
"t00000657",
"t00000658",
"t00000659",
"t00000660",
"t00000662"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00000662",
"t00000663"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00000662",
"t00000663"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000664"
      ],
      "module_path": "model.layers.11.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000664"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00000662",
"t00000665"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00000662",
"t00000665"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000666"
      ],
      "module_path": "model.layers.11.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000666"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00000662",
"t00000667"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00000662",
"t00000667"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000668"
      ],
      "module_path": "model.layers.11.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000668"
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
81
      ],
      "external_input_tensor_ids": [
"t00000664",
"t00000666",
"t00000668",
"t00000023",
"t00000680",
"t00000682",
"t00000053",
"t00000057",
"t00000723"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00000664",
"t00000669",
"t00000666",
"t00000671",
"t00000668",
"t00000673",
"t00000023",
"t00000675",
"t00000676",
"t00000680",
"t00000683",
"t00000682",
"t00000685",
"t00000670",
"t00000684",
"t00000689",
"t00000690",
"t00000688",
"t00000691",
"t00000686",
"t00000687",
"t00000692",
"t00000672",
"t00000696",
"t00000697",
"t00000695",
"t00000698",
"t00000694",
"t00000699",
"t00000700",
"t00000693",
"t00000701",
"t00000702",
"t00000703",
"t00000053",
"t00000704",
"t00000705",
"t00000706",
"t00000674",
"t00000707",
"t00000708",
"t00000709",
"t00000057",
"t00000711",
"t00000712",
"t00000713",
"t00000714",
"t00000715",
"t00000710",
"t00000717",
"t00000718",
"t00000719",
"t00000720",
"t00000721",
"t00000723",
"t00000722",
"t00000716",
"t00000725",
"t00000724",
"t00000726",
"t00000727",
"t00000728",
"t00000729",
"t00000730",
"t00000731"
      ],
      "last_event_op_index": 81,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00000677"
      ],
      "module_path": "model.layers.11.self_attn",
      "module_relative_path": "self_attn",
      "module_type": "VisiPrunerLlamaAttention",
      "op_count": 62,
      "op_counts": {
"add.Tensor": 5,
"any.default": 1,
"cat.default": 2,
"contiguous.default": 2,
"cosine_similarity.default": 1,
"div.Tensor": 1,
"dropout.default": 1,
"eq.Scalar": 1,
"gt.Scalar": 1,
"index.Tensor": 2,
"is_nonzero.default": 2,
"item.default": 2,
"lt.Scalar": 1,
"matmul.default": 2,
"mul.Tensor": 5,
"neg.default": 2,
"permute.default": 1,
"reshape.default": 1,
"select.int": 6,
"slice.Tensor": 5,
"softmax.int": 1,
"squeeze.dim": 1,
"sub.Tensor": 1,
"to.dtype": 1,
"transpose.int": 5,
"unsqueeze.default": 5,
"view.default": 4
      },
      "output_tensor_ids": [
"t00000669",
"t00000670",
"t00000671",
"t00000672",
"t00000673",
"t00000674",
"t00000675",
"t00000676",
"t00000677",
"t00000683",
"t00000684",
"t00000685",
"t00000686",
"t00000687",
"t00000688",
"t00000689",
"t00000690",
"t00000691",
"t00000692",
"t00000693",
"t00000694",
"t00000695",
"t00000696",
"t00000697",
"t00000698",
"t00000699",
"t00000700",
"t00000701",
"t00000702",
"t00000703",
"t00000704",
"t00000705",
"t00000706",
"t00000707",
"t00000708",
"t00000709",
"t00000710",
"t00000711",
"t00000712",
"t00000713",
"t00000714",
"t00000715",
"t00000716",
"t00000717",
"t00000718",
"t00000719",
"t00000720",
"t00000721",
"t00000722",
"t00000724",
"t00000725",
"t00000726",
"t00000727",
"t00000728",
"t00000729",
"t00000730",
"t00000731"
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
"t00000677",
"t00000679",
"t00000681"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00000677",
"t00000678",
"t00000679",
"t00000680",
"t00000681",
"t00000682"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.11.self_attn.rotary_emb",
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
"t00000678",
"t00000680",
"t00000682"
      ]
    },
    {
      "event_op_indices": [
82
      ],
      "external_input_tensor_ids": [
"t00000710",
"t00000732"
      ],
      "first_event_op_index": 82,
      "input_tensor_ids": [
"t00000710",
"t00000732"
      ],
      "last_event_op_index": 82,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000733"
      ],
      "module_path": "model.layers.11.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000733"
      ]
    },
    {
      "event_op_indices": [
83,
97
      ],
      "external_input_tensor_ids": [
"t00000653",
"t00000733",
"t00000751"
      ],
      "first_event_op_index": 83,
      "input_tensor_ids": [
"t00000653",
"t00000733",
"t00000734",
"t00000751"
      ],
      "last_event_op_index": 97,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00000752"
      ],
      "module_path": "model.layers.11",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00000734",
"t00000752"
      ]
    },
    {
      "event_op_indices": [
84,
85,
86,
87,
88,
89,
90,
91
      ],
      "external_input_tensor_ids": [
"t00000734",
"t00000742"
      ],
      "first_event_op_index": 84,
      "input_tensor_ids": [
"t00000734",
"t00000735",
"t00000736",
"t00000737",
"t00000738",
"t00000739",
"t00000740",
"t00000742",
"t00000741"
      ],
      "last_event_op_index": 91,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000743"
      ],
      "module_path": "model.layers.11.post_attention_layernorm",
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
"t00000735",
"t00000736",
"t00000737",
"t00000738",
"t00000739",
"t00000740",
"t00000741",
"t00000743"
      ]
    },
    {
      "event_op_indices": [
92
      ],
      "external_input_tensor_ids": [
"t00000743",
"t00000744"
      ],
      "first_event_op_index": 92,
      "input_tensor_ids": [
"t00000743",
"t00000744"
      ],
      "last_event_op_index": 92,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000745"
      ],
      "module_path": "model.layers.11.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000745"
      ]
    },
    {
      "event_op_indices": [
93
      ],
      "external_input_tensor_ids": [
"t00000745"
      ],
      "first_event_op_index": 93,
      "input_tensor_ids": [
"t00000745"
      ],
      "last_event_op_index": 93,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00000746"
      ],
      "module_path": "model.layers.11.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00000746"
      ]
    },
    {
      "event_op_indices": [
94
      ],
      "external_input_tensor_ids": [
"t00000743",
"t00000747"
      ],
      "first_event_op_index": 94,
      "input_tensor_ids": [
"t00000743",
"t00000747"
      ],
      "last_event_op_index": 94,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000748"
      ],
      "module_path": "model.layers.11.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000748"
      ]
    },
    {
      "event_op_indices": [
95
      ],
      "external_input_tensor_ids": [
"t00000746",
"t00000748"
      ],
      "first_event_op_index": 95,
      "input_tensor_ids": [
"t00000746",
"t00000748"
      ],
      "last_event_op_index": 95,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00000749"
      ],
      "module_path": "model.layers.11.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00000749"
      ]
    },
    {
      "event_op_indices": [
96
      ],
      "external_input_tensor_ids": [
"t00000749",
"t00000750"
      ],
      "first_event_op_index": 96,
      "input_tensor_ids": [
"t00000749",
"t00000750"
      ],
      "last_event_op_index": 96,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000751"
      ],
      "module_path": "model.layers.11.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000751"
      ]
    }
  ],
  "op_counts": {
    "add.Tensor": 9,
    "any.default": 1,
    "cat.default": 2,
    "contiguous.default": 2,
    "cosine_similarity.default": 1,
    "div.Tensor": 1,
    "dropout.default": 1,
    "eq.Scalar": 1,
    "gt.Scalar": 2,
    "index.Tensor": 2,
    "is_nonzero.default": 3,
    "item.default": 4,
    "linear.default": 7,
    "lt.Scalar": 1,
    "matmul.default": 2,
    "mean.dim": 2,
    "mul.Tensor": 10,
    "neg.default": 2,
    "permute.default": 1,
    "pow.Tensor_Scalar": 2,
    "reshape.default": 1,
    "rsqrt.default": 2,
    "select.int": 6,
    "silu.default": 1,
    "slice.Tensor": 7,
    "softmax.int": 1,
    "squeeze.dim": 1,
    "sub.Tensor": 1,
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
    "kv_len": 624,
    "seq": 624,
    "tail_start": null,
    "visual_end": null,
    "visual_start": null
  },
  "past_len": 0,
  "phase": "prefill",
  "priority": "P0",
  "q_len": 624,
  "row_count": 97,
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
    "edge_count": 107,
    "external_input_tensor_ids": [
      "t00000653",
      "t00000661",
      "t00000663",
      "t00000665",
      "t00000667",
      "t00000023",
      "t00000679",
      "t00000681",
      "t00000053",
      "t00000057",
      "t00000723",
      "t00000732",
      "t00000742",
      "t00000744",
      "t00000747",
      "t00000750"
    ],
    "final_output_tensor_ids": [
      "t00000752"
    ],
    "op_count": 97
  },
  "token_state": "full_visual",
  "visipruner_role": "middle_probe"
}""")
EXPECTED_STAGES = DISPATCH_FEATURES["expected_stages"]

# - input_rmsnorm: hidden_states: [624, 4096] -> normalized [624, 4096]
# - qkv_projection: Q/K/V projection: [624, 4096] -> [32, 624, 128]
# - rope: see dispatch evidence for exact tensor roles
# - attention: attention scores: [32, 624, 624]
# - visipruner_similarity_check: see dispatch evidence for exact tensor roles
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
