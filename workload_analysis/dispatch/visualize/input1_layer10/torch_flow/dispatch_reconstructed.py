#!/usr/bin/env python3
"""Torch reconstruction scaffold for input1_layer10, derived from dispatch evidence.

This file preserves the original layer's dispatch-derived process and tensor
roles. It is meant for reading, adaptation, and cross-checking against the
dispatch CSV; use `toy_tensor_compute.py` for a runnable small-shape version.
"""

from __future__ import annotations

import json
import torch
import torch.nn.functional as F


EVENT_ID = 'input1_layer10'
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
  "t00000570"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000571"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000572"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000573"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000574"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000575"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000588",
  "t00000593"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000594"
]
      },
      {
"event_op_index": 46,
"input_tensor_ids": [
  "t00000595",
  "t00000600"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000601"
]
      },
      {
"event_op_index": 47,
"input_tensor_ids": [
  "t00000601"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 128, 624], dtype=float16",
"output_tensor_ids": [
  "t00000602"
]
      },
      {
"event_op_index": 48,
"input_tensor_ids": [
  "t00000594",
  "t00000602"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000603"
]
      },
      {
"event_op_index": 49,
"input_tensor_ids": [
  "t00000603"
],
"op_name": "div.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000604"
]
      },
      {
"event_op_index": 50,
"input_tensor_ids": [
  "t00000604",
  "t00000053"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000605"
]
      },
      {
"event_op_index": 51,
"input_tensor_ids": [
  "t00000605"
],
"op_name": "softmax.int",
"output": "shape=[1, 32, 624, 624], dtype=float32",
"output_tensor_ids": [
  "t00000606"
]
      },
      {
"event_op_index": 53,
"input_tensor_ids": [
  "t00000607"
],
"op_name": "dropout.default",
"output": "shape=[1, 32, 624, 624], dtype=float16",
"output_tensor_ids": [
  "t00000607"
]
      },
      {
"event_op_index": 54,
"input_tensor_ids": [
  "t00000607",
  "t00000575"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000608"
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
  "t00000607",
  "t00000575"
],
"op_name": "matmul.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000608"
]
      },
      {
"event_op_index": 56,
"input_tensor_ids": [
  "t00000609"
],
"op_name": "contiguous.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000610"
]
      },
      {
"event_op_index": 57,
"input_tensor_ids": [
  "t00000610"
],
"op_name": "reshape.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000611"
]
      },
      {
"event_op_index": 70,
"input_tensor_ids": [
  "t00000621"
],
"op_name": "contiguous.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000622"
]
      },
      {
"event_op_index": 82,
"input_tensor_ids": [
  "t00000611",
  "t00000633"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000634"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00000554",
  "t00000634"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000635"
]
      },
      {
"event_op_index": 96,
"input_tensor_ids": [
  "t00000650",
  "t00000651"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000652"
]
      },
      {
"event_op_index": 97,
"input_tensor_ids": [
  "t00000635",
  "t00000652"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000653"
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
  "t00000554"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000555"
]
      },
      {
"event_op_index": 2,
"input_tensor_ids": [
  "t00000555"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000556"
]
      },
      {
"event_op_index": 3,
"input_tensor_ids": [
  "t00000556"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000557"
]
      },
      {
"event_op_index": 4,
"input_tensor_ids": [
  "t00000557"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000558"
]
      },
      {
"event_op_index": 5,
"input_tensor_ids": [
  "t00000558"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000559"
]
      },
      {
"event_op_index": 6,
"input_tensor_ids": [
  "t00000555",
  "t00000559"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000560"
]
      },
      {
"event_op_index": 7,
"input_tensor_ids": [
  "t00000560"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000561"
]
      },
      {
"event_op_index": 8,
"input_tensor_ids": [
  "t00000562",
  "t00000561"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000563"
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
  "t00000611",
  "t00000633"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000634"
]
      },
      {
"event_op_index": 83,
"input_tensor_ids": [
  "t00000554",
  "t00000634"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000635"
]
      },
      {
"event_op_index": 84,
"input_tensor_ids": [
  "t00000635"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000636"
]
      },
      {
"event_op_index": 85,
"input_tensor_ids": [
  "t00000636"
],
"op_name": "pow.Tensor_Scalar",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000637"
]
      },
      {
"event_op_index": 86,
"input_tensor_ids": [
  "t00000637"
],
"op_name": "mean.dim",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000638"
]
      },
      {
"event_op_index": 87,
"input_tensor_ids": [
  "t00000638"
],
"op_name": "add.Tensor",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000639"
]
      },
      {
"event_op_index": 88,
"input_tensor_ids": [
  "t00000639"
],
"op_name": "rsqrt.default",
"output": "shape=[1, 624, 1], dtype=float32",
"output_tensor_ids": [
  "t00000640"
]
      },
      {
"event_op_index": 89,
"input_tensor_ids": [
  "t00000636",
  "t00000640"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float32",
"output_tensor_ids": [
  "t00000641"
]
      },
      {
"event_op_index": 90,
"input_tensor_ids": [
  "t00000641"
],
"op_name": "to.dtype",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000642"
]
      },
      {
"event_op_index": 91,
"input_tensor_ids": [
  "t00000643",
  "t00000642"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000644"
]
      },
      {
"event_op_index": 92,
"input_tensor_ids": [
  "t00000644",
  "t00000645"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000646"
]
      },
      {
"event_op_index": 93,
"input_tensor_ids": [
  "t00000646"
],
"op_name": "silu.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000647"
]
      },
      {
"event_op_index": 94,
"input_tensor_ids": [
  "t00000644",
  "t00000648"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000649"
]
      },
      {
"event_op_index": 95,
"input_tensor_ids": [
  "t00000647",
  "t00000649"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 624, 11008], dtype=float16",
"output_tensor_ids": [
  "t00000650"
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
  "t00000563",
  "t00000564"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000565"
]
      },
      {
"event_op_index": 10,
"input_tensor_ids": [
  "t00000563",
  "t00000566"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000567"
]
      },
      {
"event_op_index": 11,
"input_tensor_ids": [
  "t00000563",
  "t00000568"
],
"op_name": "linear.default",
"output": "shape=[1, 624, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000569"
]
      },
      {
"event_op_index": 12,
"input_tensor_ids": [
  "t00000565"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000570"
]
      },
      {
"event_op_index": 13,
"input_tensor_ids": [
  "t00000570"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000571"
]
      },
      {
"event_op_index": 14,
"input_tensor_ids": [
  "t00000567"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000572"
]
      },
      {
"event_op_index": 15,
"input_tensor_ids": [
  "t00000572"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000573"
]
      },
      {
"event_op_index": 16,
"input_tensor_ids": [
  "t00000569"
],
"op_name": "view.default",
"output": "shape=[1, 624, 32, 128], dtype=float16",
"output_tensor_ids": [
  "t00000574"
]
      },
      {
"event_op_index": 17,
"input_tensor_ids": [
  "t00000574"
],
"op_name": "transpose.int",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000575"
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
  "t00000577"
],
"op_name": "add.Tensor",
"output": "shape=[], dtype=int64",
"output_tensor_ids": [
  "t00000578"
]
      },
      {
"event_op_index": 24,
"input_tensor_ids": [
  "t00000580"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000581"
]
      },
      {
"event_op_index": 27,
"input_tensor_ids": [
  "t00000582"
],
"op_name": "slice.Tensor",
"output": "shape=[624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000583"
]
      },
      {
"event_op_index": 29,
"input_tensor_ids": [
  "t00000581",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000584"
]
      },
      {
"event_op_index": 30,
"input_tensor_ids": [
  "t00000584"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000585"
]
      },
      {
"event_op_index": 31,
"input_tensor_ids": [
  "t00000583",
  "t00000023"
],
"op_name": "index.Tensor",
"output": "shape=[1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000586"
]
      },
      {
"event_op_index": 32,
"input_tensor_ids": [
  "t00000586"
],
"op_name": "unsqueeze.default",
"output": "shape=[1, 1, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000587"
]
      },
      {
"event_op_index": 33,
"input_tensor_ids": [
  "t00000571",
  "t00000585"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000588"
]
      },
      {
"event_op_index": 34,
"input_tensor_ids": [
  "t00000571"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000589"
]
      },
      {
"event_op_index": 35,
"input_tensor_ids": [
  "t00000571"
],
"op_name": "slice.Tensor",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000590"
]
      },
      {
"event_op_index": 36,
"input_tensor_ids": [
  "t00000590"
],
"op_name": "neg.default",
"output": "shape=[1, 32, 624, 64], dtype=float16",
"output_tensor_ids": [
  "t00000591"
]
      },
      {
"event_op_index": 37,
"input_tensor_ids": [
  "t00000591",
  "t00000589"
],
"op_name": "cat.default",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000592"
]
      },
      {
"event_op_index": 38,
"input_tensor_ids": [
  "t00000592",
  "t00000587"
],
"op_name": "mul.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000593"
]
      },
      {
"event_op_index": 39,
"input_tensor_ids": [
  "t00000588",
  "t00000593"
],
"op_name": "add.Tensor",
"output": "shape=[1, 32, 624, 128], dtype=float16",
"output_tensor_ids": [
  "t00000594"
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
  "t00000578"
],
"op_name": "gt.Scalar",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00000579"
]
      },
      {
"event_op_index": 22,
"input_tensor_ids": [
  "t00000579"
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
  "t00000612"
]
      },
      {
"event_op_index": 59,
"input_tensor_ids": [
  "t00000612"
],
"op_name": "is_nonzero.default",
"output": "True",
"output_tensor_ids": []
      },
      {
"event_op_index": 64,
"input_tensor_ids": [
  "t00000616"
],
"op_name": "is_nonzero.default",
"output": "True",
"output_tensor_ids": []
      },
      {
"event_op_index": 75,
"input_tensor_ids": [
  "t00000626",
  "t00000625"
],
"op_name": "sub.Tensor",
"output": "shape=[1, 576, 4096], dtype=float16",
"output_tensor_ids": [
  "t00000627"
]
      },
      {
"event_op_index": 77,
"input_tensor_ids": [
  "t00000627",
  "t00000628"
],
"op_name": "cosine_similarity.default",
"output": "shape=[1, 576], dtype=float16",
"output_tensor_ids": [
  "t00000629"
]
      },
      {
"event_op_index": 80,
"input_tensor_ids": [
  "t00000631"
],
"op_name": "any.default",
"output": "shape=[], dtype=bool",
"output_tensor_ids": [
  "t00000632"
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
  "event_id": "input1_layer10",
  "input_id": 1,
  "kv_len": 624,
  "layer_id": 10,
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
"t00000554",
"t00000562"
      ],
      "first_event_op_index": 1,
      "input_tensor_ids": [
"t00000554",
"t00000555",
"t00000556",
"t00000557",
"t00000558",
"t00000559",
"t00000560",
"t00000562",
"t00000561"
      ],
      "last_event_op_index": 8,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000563"
      ],
      "module_path": "model.layers.10.input_layernorm",
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
"t00000555",
"t00000556",
"t00000557",
"t00000558",
"t00000559",
"t00000560",
"t00000561",
"t00000563"
      ]
    },
    {
      "event_op_indices": [
9
      ],
      "external_input_tensor_ids": [
"t00000563",
"t00000564"
      ],
      "first_event_op_index": 9,
      "input_tensor_ids": [
"t00000563",
"t00000564"
      ],
      "last_event_op_index": 9,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000565"
      ],
      "module_path": "model.layers.10.self_attn.q_proj",
      "module_relative_path": "self_attn.q_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000565"
      ]
    },
    {
      "event_op_indices": [
10
      ],
      "external_input_tensor_ids": [
"t00000563",
"t00000566"
      ],
      "first_event_op_index": 10,
      "input_tensor_ids": [
"t00000563",
"t00000566"
      ],
      "last_event_op_index": 10,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000567"
      ],
      "module_path": "model.layers.10.self_attn.k_proj",
      "module_relative_path": "self_attn.k_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000567"
      ]
    },
    {
      "event_op_indices": [
11
      ],
      "external_input_tensor_ids": [
"t00000563",
"t00000568"
      ],
      "first_event_op_index": 11,
      "input_tensor_ids": [
"t00000563",
"t00000568"
      ],
      "last_event_op_index": 11,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000569"
      ],
      "module_path": "model.layers.10.self_attn.v_proj",
      "module_relative_path": "self_attn.v_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000569"
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
"t00000565",
"t00000567",
"t00000569",
"t00000023",
"t00000581",
"t00000583",
"t00000053",
"t00000057",
"t00000624"
      ],
      "first_event_op_index": 12,
      "input_tensor_ids": [
"t00000565",
"t00000570",
"t00000567",
"t00000572",
"t00000569",
"t00000574",
"t00000023",
"t00000576",
"t00000577",
"t00000581",
"t00000584",
"t00000583",
"t00000586",
"t00000571",
"t00000585",
"t00000590",
"t00000591",
"t00000589",
"t00000592",
"t00000587",
"t00000588",
"t00000593",
"t00000573",
"t00000597",
"t00000598",
"t00000596",
"t00000599",
"t00000595",
"t00000600",
"t00000601",
"t00000594",
"t00000602",
"t00000603",
"t00000604",
"t00000053",
"t00000605",
"t00000606",
"t00000607",
"t00000575",
"t00000608",
"t00000609",
"t00000610",
"t00000057",
"t00000612",
"t00000613",
"t00000614",
"t00000615",
"t00000616",
"t00000611",
"t00000618",
"t00000619",
"t00000620",
"t00000621",
"t00000622",
"t00000624",
"t00000623",
"t00000617",
"t00000626",
"t00000625",
"t00000627",
"t00000628",
"t00000629",
"t00000630",
"t00000631",
"t00000632"
      ],
      "last_event_op_index": 81,
      "module_class": "llava.model.language_model.custom_modeling_llama.VisiPrunerLlamaAttention",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "614",
      "module_output_tensor_ids": [
"t00000578"
      ],
      "module_path": "model.layers.10.self_attn",
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
"t00000570",
"t00000571",
"t00000572",
"t00000573",
"t00000574",
"t00000575",
"t00000576",
"t00000577",
"t00000578",
"t00000584",
"t00000585",
"t00000586",
"t00000587",
"t00000588",
"t00000589",
"t00000590",
"t00000591",
"t00000592",
"t00000593",
"t00000594",
"t00000595",
"t00000596",
"t00000597",
"t00000598",
"t00000599",
"t00000600",
"t00000601",
"t00000602",
"t00000603",
"t00000604",
"t00000605",
"t00000606",
"t00000607",
"t00000608",
"t00000609",
"t00000610",
"t00000611",
"t00000612",
"t00000613",
"t00000614",
"t00000615",
"t00000616",
"t00000617",
"t00000618",
"t00000619",
"t00000620",
"t00000621",
"t00000622",
"t00000623",
"t00000625",
"t00000626",
"t00000627",
"t00000628",
"t00000629",
"t00000630",
"t00000631",
"t00000632"
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
"t00000578",
"t00000580",
"t00000582"
      ],
      "first_event_op_index": 21,
      "input_tensor_ids": [
"t00000578",
"t00000579",
"t00000580",
"t00000581",
"t00000582",
"t00000583"
      ],
      "last_event_op_index": 28,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRotaryEmbedding",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "175",
      "module_output_tensor_ids": [],
      "module_path": "model.layers.10.self_attn.rotary_emb",
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
"t00000579",
"t00000581",
"t00000583"
      ]
    },
    {
      "event_op_indices": [
82
      ],
      "external_input_tensor_ids": [
"t00000611",
"t00000633"
      ],
      "first_event_op_index": 82,
      "input_tensor_ids": [
"t00000611",
"t00000633"
      ],
      "last_event_op_index": 82,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000634"
      ],
      "module_path": "model.layers.10.self_attn.o_proj",
      "module_relative_path": "self_attn.o_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000634"
      ]
    },
    {
      "event_op_indices": [
83,
97
      ],
      "external_input_tensor_ids": [
"t00000554",
"t00000634",
"t00000652"
      ],
      "first_event_op_index": 83,
      "input_tensor_ids": [
"t00000554",
"t00000634",
"t00000635",
"t00000652"
      ],
      "last_event_op_index": 97,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaDecoderLayer",
      "module_forward_file": "/workspace/VisiPrune/workload_analysis/dispatch/tools/visipruner_filtered_dispatch_profile.py",
      "module_forward_lineno": "881",
      "module_output_tensor_ids": [
"t00000653"
      ],
      "module_path": "model.layers.10",
      "module_relative_path": "",
      "module_type": "LlamaDecoderLayer",
      "op_count": 2,
      "op_counts": {
"add.Tensor": 2
      },
      "output_tensor_ids": [
"t00000635",
"t00000653"
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
"t00000635",
"t00000643"
      ],
      "first_event_op_index": 84,
      "input_tensor_ids": [
"t00000635",
"t00000636",
"t00000637",
"t00000638",
"t00000639",
"t00000640",
"t00000641",
"t00000643",
"t00000642"
      ],
      "last_event_op_index": 91,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaRMSNorm",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "139",
      "module_output_tensor_ids": [
"t00000644"
      ],
      "module_path": "model.layers.10.post_attention_layernorm",
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
"t00000636",
"t00000637",
"t00000638",
"t00000639",
"t00000640",
"t00000641",
"t00000642",
"t00000644"
      ]
    },
    {
      "event_op_indices": [
92
      ],
      "external_input_tensor_ids": [
"t00000644",
"t00000645"
      ],
      "first_event_op_index": 92,
      "input_tensor_ids": [
"t00000644",
"t00000645"
      ],
      "last_event_op_index": 92,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000646"
      ],
      "module_path": "model.layers.10.mlp.gate_proj",
      "module_relative_path": "mlp.gate_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000646"
      ]
    },
    {
      "event_op_indices": [
93
      ],
      "external_input_tensor_ids": [
"t00000646"
      ],
      "first_event_op_index": 93,
      "input_tensor_ids": [
"t00000646"
      ],
      "last_event_op_index": 93,
      "module_class": "torch.nn.modules.activation.SiLU",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/activation.py",
      "module_forward_lineno": "471",
      "module_output_tensor_ids": [
"t00000647"
      ],
      "module_path": "model.layers.10.mlp.act_fn",
      "module_relative_path": "mlp.act_fn",
      "module_type": "SiLU",
      "op_count": 1,
      "op_counts": {
"silu.default": 1
      },
      "output_tensor_ids": [
"t00000647"
      ]
    },
    {
      "event_op_indices": [
94
      ],
      "external_input_tensor_ids": [
"t00000644",
"t00000648"
      ],
      "first_event_op_index": 94,
      "input_tensor_ids": [
"t00000644",
"t00000648"
      ],
      "last_event_op_index": 94,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000649"
      ],
      "module_path": "model.layers.10.mlp.up_proj",
      "module_relative_path": "mlp.up_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000649"
      ]
    },
    {
      "event_op_indices": [
95
      ],
      "external_input_tensor_ids": [
"t00000647",
"t00000649"
      ],
      "first_event_op_index": 95,
      "input_tensor_ids": [
"t00000647",
"t00000649"
      ],
      "last_event_op_index": 95,
      "module_class": "llava.model.language_model.custom_modeling_llama.LlamaMLP",
      "module_forward_file": "/workspace/VisiPrune/repo/llava/model/language_model/custom_modeling_llama.py",
      "module_forward_lineno": "277",
      "module_output_tensor_ids": [
"t00000650"
      ],
      "module_path": "model.layers.10.mlp",
      "module_relative_path": "mlp",
      "module_type": "LlamaMLP",
      "op_count": 1,
      "op_counts": {
"mul.Tensor": 1
      },
      "output_tensor_ids": [
"t00000650"
      ]
    },
    {
      "event_op_indices": [
96
      ],
      "external_input_tensor_ids": [
"t00000650",
"t00000651"
      ],
      "first_event_op_index": 96,
      "input_tensor_ids": [
"t00000650",
"t00000651"
      ],
      "last_event_op_index": 96,
      "module_class": "torch.nn.modules.linear.Linear",
      "module_forward_file": "/opt/conda/envs/cu132/lib/python3.12/site-packages/torch/nn/modules/linear.py",
      "module_forward_lineno": "130",
      "module_output_tensor_ids": [
"t00000652"
      ],
      "module_path": "model.layers.10.mlp.down_proj",
      "module_relative_path": "mlp.down_proj",
      "module_type": "Linear",
      "op_count": 1,
      "op_counts": {
"linear.default": 1
      },
      "output_tensor_ids": [
"t00000652"
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
      "t00000554",
      "t00000562",
      "t00000564",
      "t00000566",
      "t00000568",
      "t00000023",
      "t00000580",
      "t00000582",
      "t00000053",
      "t00000057",
      "t00000624",
      "t00000633",
      "t00000643",
      "t00000645",
      "t00000648",
      "t00000651"
    ],
    "final_output_tensor_ids": [
      "t00000653"
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
