#!/usr/bin/env python3
"""Build tensor-layout examples and graphs for filtered dispatch events.

The filtered dispatch trace records ATen calls and tensor metadata, but not
stable Python tensor object identities. This script therefore builds a
shape-level computation graph: op nodes are exact dispatch operations, tensor
nodes are derived from the recorded tensor layouts, and lineage is inferred
from the most recent matching tensor layout when possible.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOAD_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DISPATCH_DIR = (
    WORKLOAD_DIR
    / "dispatch/profiles/filtered_dispatch_visipruner_full_32tok"
)
DEFAULT_OUTPUT_DIR = WORKLOAD_DIR / "dispatch/visualize"

ROLE_COLORS = {
    "shallow_or_boundary": "#e8f4ff",
    "middle_probe": "#fff5d7",
    "middle_select": "#ffe8cc",
    "deep_check": "#eaf7e7",
    "boundary_before_prune": "#f5e8ff",
    "boundary_after_prune": "#f5e8ff",
    "decode_prune_effect": "#e9f0ff",
}

OP_COLORS = {
    "linear": "#dceeff",
    "matmul": "#dceeff",
    "softmax": "#ffe0e0",
    "cosine_similarity": "#e7f7df",
    "linalg_vector_norm": "#e7f7df",
    "where": "#ffe8cc",
    "index": "#f6ecff",
    "slice": "#f6ecff",
    "cat": "#f2f2f2",
}


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TensorLayout":
        return cls(
            shape=tuple(int(dim) for dim in record.get("shape", [])),
            dtype=str(record.get("dtype", "")),
            device=str(record.get("device", "")),
            requires_grad=bool(record.get("requires_grad", False)),
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
        }

    def label(self) -> str:
        shape = "[" + ", ".join(str(dim) for dim in self.shape) + "]"
        grad = ", grad" if self.requires_grad else ""
        return f"{shape}\\n{self.dtype} {self.device}{grad}"

    def short(self) -> str:
        shape = "[" + ",".join(str(dim) for dim in self.shape) + "]"
        return f"{shape} {self.dtype}"

    def key(self) -> tuple[tuple[int, ...], str, str, bool]:
        return (self.shape, self.dtype, self.device, self.requires_grad)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def op_family(op_name: str) -> str:
    return op_name.split(".", 1)[0]


def op_color(op_name: str) -> str:
    for prefix, color in OP_COLORS.items():
        if op_name.startswith(prefix):
            return color
    return "#ffffff"


def flatten_tensors(value: Any, path: str = "value") -> list[tuple[str, TensorLayout]]:
    if isinstance(value, dict) and value.get("type") == "Tensor":
        return [(path, TensorLayout.from_record(value))]
    if isinstance(value, list):
        tensors: list[tuple[str, TensorLayout]] = []
        for idx, item in enumerate(value):
            tensors.extend(flatten_tensors(item, f"{path}[{idx}]"))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for key, item in value.items():
            tensors.extend(flatten_tensors(item, f"{path}.{key}"))
        return tensors
    return []


def compact_value(value: Any, max_chars: int = 180) -> str:
    if isinstance(value, dict) and value.get("type") == "Tensor":
        return f"Tensor(shape={value.get('shape')}, dtype={value.get('dtype')})"
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def event_role_note(row: dict[str, str]) -> list[str]:
    role = row.get("visipruner_role", "")
    q_len = int(row.get("q_len") or 0)
    kv_len = int(row.get("kv_len") or 0)
    selection = row.get("selection_result", "")
    token_state = row.get("token_state", "")
    notes = [
        f"phase={row.get('phase')}, q_len={q_len}, kv_len={kv_len}, token_state={token_state}",
        f"priority={row.get('priority')}, role={role}, selection_result={selection or 'n/a'}",
    ]
    if "middle_probe" in role:
        notes.append(
            "Middle-stage probe: attention/hidden-state features are computed, "
            "but no visual token index tensor is emitted for this layer."
        )
    if "middle_select" in role:
        notes.append(
            "Middle-stage selection: visual-token features are compared with the "
            "last token, vector norms are thresholded, and the recorded trace "
            "emits a selected visual-token index tensor."
        )
    if "deep_check" in role:
        notes.append(
            "Deep-exit check: the retained visual-token window is compared with "
            "the current last token to decide whether visual tokens can be "
            "dropped after this layer."
        )
    if "boundary_after_prune" in role:
        notes.append("Boundary-after-prune event: the layer executes after q_len/kv_len shrink.")
    if "decode_prune_effect" in role:
        notes.append("Decode representative: q_len=1 while the KV-cache length reflects the pruning regime.")
    if "shallow" in role:
        notes.append("Shallow/control-boundary representative before physical visual-token removal.")
    return notes


def build_graph(
    event: dict[str, str],
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[tuple[str, str, str]]]:
    tensor_layouts: dict[str, dict[str, Any]] = {}
    operations: list[dict[str, Any]] = []
    edges: list[tuple[str, str, str]] = []
    last_output_by_layout: dict[tuple[tuple[int, ...], str, str, bool], str] = {}
    external_counter = 0

    def add_tensor(name: str, layout: TensorLayout, origin: str) -> str:
        if name not in tensor_layouts:
            payload = layout.as_json()
            payload["origin"] = origin
            tensor_layouts[name] = payload
        return name

    for raw in sorted(rows, key=lambda row: int(row["event_op_index"])):
        idx = int(raw["event_op_index"])
        op_id = f"op_{idx:03d}"
        op_name = raw["op_name"]
        args = safe_json_loads(raw.get("args", "null"))
        kwargs = safe_json_loads(raw.get("kwargs", "null"))
        outputs = safe_json_loads(raw.get("outputs", "null"))
        input_refs: list[dict[str, str]] = []

        for source_name, tensor_root in (("args", args), ("kwargs", kwargs)):
            for path, layout in flatten_tensors(tensor_root, source_name):
                layout_key = layout.key()
                matched = last_output_by_layout.get(layout_key)
                if matched is None:
                    external_counter += 1
                    tensor_name = add_tensor(
                        f"input_{external_counter:03d}",
                        layout,
                        f"{op_id}:{path}",
                    )
                else:
                    tensor_name = matched
                input_refs.append({"name": tensor_name, "path": path})
                edges.append((tensor_name, op_id, path))

        output_refs: list[dict[str, str]] = []
        for out_idx, (path, layout) in enumerate(flatten_tensors(outputs, "outputs")):
            tensor_name = add_tensor(
                f"{op_id}_out{out_idx}",
                layout,
                f"{op_id}:{path}",
            )
            last_output_by_layout[layout.key()] = tensor_name
            output_refs.append({"name": tensor_name, "path": path})
            edges.append((op_id, tensor_name, path))

        operations.append(
            {
                "id": op_id,
                "index": idx,
                "op_name": op_name,
                "op_schema": raw["op_schema"],
                "inputs": input_refs,
                "outputs": output_refs,
                "args_summary": compact_value(args),
                "kwargs_summary": compact_value(kwargs),
                "outputs_summary": compact_value(outputs),
            }
        )

    return tensor_layouts, operations, edges


def computation_steps(event: dict[str, str], operations: list[dict[str, Any]]) -> list[str]:
    families = Counter(op_family(op["op_name"]) for op in operations)
    role = event.get("visipruner_role", "")
    steps = [
        "RMSNorm over hidden_states using to/pow/mean/add/rsqrt/mul.",
        "Q/K/V projection with linear ops, then view/transpose into [batch, heads, q_len, head_dim].",
        "Rotary-position preparation with select/slice/index/cat/mul/add over cos/sin layouts.",
        "Attention score path with matmul/div/add/softmax/dropout, followed by value matmul and output projection.",
        "Residual connection, post-attention RMSNorm, gated MLP (gate/up linear, silu, multiply, down linear), final residual.",
    ]
    if "middle_probe" in role:
        steps.insert(
            4,
            "VisiPrune middle probe computes cosine similarity against visual-token layouts and returns none.",
        )
    if "middle_select" in role:
        steps.insert(
            4,
            "VisiPrune middle selection computes cosine similarity and vector-norm filters, then where() emits selected visual-token indices.",
        )
    if "deep_check" in role:
        steps.insert(
            4,
            "VisiPrune deep-exit check indexes the retained 10-token visual window and tests any(cosine_similarity < threshold).",
        )
    if "decode_prune_effect" in role:
        steps.insert(
            3,
            "Decode path concatenates current K/V with cached K/V; past_len/kv_len encode the pruning regime.",
        )
    if "shallow" in role:
        steps.insert(
            3,
            "Shallow/control path applies attention-mask related slices/fills/copies without changing q_len.",
        )
    steps.append(
        "Recorded op families: "
        + ", ".join(f"{name}={count}" for name, count in sorted(families.items()))
        + "."
    )
    return steps


def python_literal(value: Any) -> str:
    return repr(value).replace("True", "True").replace("False", "False")


def write_example_py(
    path: Path,
    event: dict[str, str],
    tensor_layouts: dict[str, dict[str, Any]],
    operations: list[dict[str, Any]],
) -> None:
    steps = computation_steps(event, operations)
    metadata = {
        "event_id": event["event_id"],
        "input_id": int(event["input_id"]),
        "layer_id": int(event["layer_id"]),
        "phase": event["phase"],
        "priority": event["priority"],
        "q_len": int(event["q_len"]),
        "past_len": int(event["past_len"]),
        "kv_len": int(event["kv_len"]),
        "token_state": event["token_state"],
        "visipruner_role": event["visipruner_role"],
        "selection_result": event.get("selection_result", ""),
        "reason": event.get("reason", ""),
    }
    lines = [
        "#!/usr/bin/env python3",
        '"""Tensor-layout computation example generated from dispatch trace.',
        "",
        "This is a shape-level example: it preserves ATen op order, tensor",
        "layouts, and VisiPrune role metadata without allocating model weights.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from pprint import pprint",
        "",
        "",
        "@dataclass(frozen=True)",
        "class TensorLayout:",
        "    shape: tuple[int, ...]",
        "    dtype: str",
        "    device: str",
        "    requires_grad: bool = False",
        "",
        "",
        f"EVENT = {python_literal(metadata)}",
        "",
        "tensor_layouts = {",
    ]
    for name, layout in sorted(tensor_layouts.items()):
        lines.append(
            f"    {name!r}: TensorLayout("
            f"shape={tuple(layout['shape'])!r}, dtype={layout['dtype']!r}, "
            f"device={layout['device']!r}, requires_grad={layout['requires_grad']!r}),"
        )
    lines.extend(
        [
            "}",
            "",
            f"COMPUTATION_STEPS = {python_literal(steps)}",
            "",
            "operations = [",
        ]
    )
    for op in operations:
        payload = {
            "id": op["id"],
            "index": op["index"],
            "op_name": op["op_name"],
            "op_schema": op["op_schema"],
            "inputs": op["inputs"],
            "outputs": op["outputs"],
        }
        lines.append(f"    {python_literal(payload)},")
    lines.extend(
        [
            "]",
            "",
            "",
            "def build_shape_level_trace():",
            "    trace = []",
            "    for op in operations:",
            "        trace.append({",
            "            'op': op['op_name'],",
            "            'input_layouts': [tensor_layouts[item['name']] for item in op['inputs']],",
            "            'output_layouts': [tensor_layouts[item['name']] for item in op['outputs']],",
            "        })",
            "    return trace",
            "",
            "",
            "def main():",
            "    print(EVENT)",
            "    print('\\nComputation steps:')",
            "    for idx, step in enumerate(COMPUTATION_STEPS, 1):",
            "        print(f'{idx}. {step}')",
            "    print('\\nFirst five shape-level ATen operations:')",
            "    pprint(build_shape_level_trace()[:5])",
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def dot_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_dot(
    path: Path,
    event: dict[str, str],
    tensor_layouts: dict[str, dict[str, Any]],
    operations: list[dict[str, Any]],
    edges: list[tuple[str, str, str]],
) -> None:
    lines = [
        "digraph tensor_graph {",
        "  rankdir=LR;",
        "  graph [fontname=Helvetica, fontsize=18, labelloc=t, label="
        + dot_quote(
            f"{event['event_id']} | {event['phase']} | "
            f"{event['visipruner_role']} | q={event['q_len']} kv={event['kv_len']}"
        )
        + "];",
        "  node [fontname=Helvetica, fontsize=10];",
        "  edge [fontname=Helvetica, fontsize=8, color=\"#606060\"];",
    ]
    for name, layout in sorted(tensor_layouts.items()):
        tensor = TensorLayout(
            shape=tuple(layout["shape"]),
            dtype=layout["dtype"],
            device=layout["device"],
            requires_grad=layout["requires_grad"],
        )
        lines.append(
            f"  {dot_quote(name)} [shape=ellipse, style=filled, fillcolor=\"#f8f8f8\", "
            f"label={dot_quote(name + '\\n' + tensor.label())}];"
        )
    for op in operations:
        fill = op_color(op["op_name"])
        label = f"{op['index']:03d}\\n{op['op_name']}"
        lines.append(
            f"  {dot_quote(op['id'])} [shape=box, style=\"rounded,filled\", "
            f"fillcolor={dot_quote(fill)}, label={dot_quote(label)}];"
        )
    for src, dst, label in edges:
        lines.append(f"  {dot_quote(src)} -> {dot_quote(dst)} [label={dot_quote(label)}];")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_text(text: str, x: int, y: int, size: int = 13, weight: str = "400") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#17202a">'
        f"{html.escape(text)}</text>"
    )


def wrap_label(text: str, width: int = 96) -> list[str]:
    if len(text) <= width:
        return [text]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def write_svg(
    path: Path,
    event: dict[str, str],
    operations: list[dict[str, Any]],
    tensor_layouts: dict[str, dict[str, Any]],
) -> None:
    row_h = 54
    width = 1220
    height = 170 + max(1, len(operations)) * row_h
    role_key = event.get("visipruner_role", "").split(";", 1)[0]
    header_color = ROLE_COLORS.get(role_key, "#eef2f7")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="24" y="20" width="{width - 48}" height="96" rx="8" fill="{header_color}" stroke="#b8c2cc"/>',
        svg_text(
            f"{event['event_id']} tensor computation graph",
            44,
            52,
            size=20,
            weight="700",
        ),
        svg_text(
            f"phase={event['phase']} priority={event['priority']} role={event['visipruner_role']} "
            f"q_len={event['q_len']} past_len={event['past_len']} kv_len={event['kv_len']} "
            f"token_state={event['token_state']} selection={event.get('selection_result') or 'n/a'}",
            44,
            80,
            size=13,
        ),
        svg_text(
            "Rows are exact ATen dispatch ops; output tensor layouts are recorded from dispatch metadata.",
            44,
            102,
            size=12,
        ),
        svg_text("op", 44, 145, size=12, weight="700"),
        svg_text("dispatch op", 122, 145, size=12, weight="700"),
        svg_text("output tensor layout(s)", 430, 145, size=12, weight="700"),
        svg_text("shape-level inputs", 760, 145, size=12, weight="700"),
    ]

    y0 = 164
    for row_idx, op in enumerate(operations):
        y = y0 + row_idx * row_h
        fill = op_color(op["op_name"])
        parts.append(
            f'<rect x="30" y="{y - 28}" width="{width - 60}" height="44" rx="6" '
            f'fill="{fill}" stroke="#d6dbe1"/>'
        )
        if row_idx:
            parts.append(
                f'<line x1="70" y1="{y - row_h + 18}" x2="70" y2="{y - 30}" '
                f'stroke="#8391a1" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
        parts.append(svg_text(f"{op['index']:03d}", 44, y, size=12, weight="700"))
        parts.append(svg_text(op["op_name"], 122, y, size=13, weight="700"))
        schema_lines = wrap_label(op["op_schema"], 52)
        for line_no, line in enumerate(schema_lines[:2]):
            parts.append(svg_text(line, 122, y + 15 + line_no * 13, size=10))
        out_labels = []
        for out in op["outputs"][:3]:
            layout = tensor_layouts[out["name"]]
            tensor = TensorLayout(
                shape=tuple(layout["shape"]),
                dtype=layout["dtype"],
                device=layout["device"],
                requires_grad=layout["requires_grad"],
            )
            out_labels.append(f"{out['name']} {tensor.short()}")
        if len(op["outputs"]) > 3:
            out_labels.append(f"... +{len(op['outputs']) - 3} more")
        if not out_labels:
            out_labels = ["non-tensor output"]
        for line_no, line in enumerate(out_labels[:3]):
            parts.append(svg_text(line, 430, y + line_no * 14, size=11))
        in_labels = []
        for item in op["inputs"][:3]:
            layout = tensor_layouts[item["name"]]
            tensor = TensorLayout(
                shape=tuple(layout["shape"]),
                dtype=layout["dtype"],
                device=layout["device"],
                requires_grad=layout["requires_grad"],
            )
            in_labels.append(f"{item['path']} <- {item['name']} {tensor.short()}")
        if len(op["inputs"]) > 3:
            in_labels.append(f"... +{len(op['inputs']) - 3} more")
        if not in_labels:
            in_labels = ["no tensor input"]
        for line_no, line in enumerate(in_labels[:3]):
            parts.append(svg_text(line, 760, y + line_no * 14, size=11))

    parts.insert(
        1,
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="5" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#8391a1"/></marker></defs>',
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_summary_md(
    path: Path,
    event: dict[str, str],
    operations: list[dict[str, Any]],
    tensor_layouts: dict[str, dict[str, Any]],
) -> None:
    families = Counter(op_family(op["op_name"]) for op in operations)
    notes = event_role_note(event)
    lines = [
        f"# {event['event_id']}",
        "",
        "## Dispatch Context",
        "",
        f"- phase: `{event['phase']}`",
        f"- priority: `{event['priority']}`",
        f"- layer_id: `{event['layer_id']}`",
        f"- q_len/past_len/kv_len: `{event['q_len']}/{event['past_len']}/{event['kv_len']}`",
        f"- token_state: `{event['token_state']}`",
        f"- role: `{event['visipruner_role']}`",
        f"- selection_result: `{event.get('selection_result') or 'n/a'}`",
        "",
        "## Role Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `example.py`: tensor_layouts-based computation example.",
            "- `tensor_layouts.json`: machine-readable layouts, ops, and graph edges.",
            "- `tensor_graph.dot`: DOT graph source.",
            "- `tensor_graph.svg`: standalone rendered tensor computation graph.",
            "",
            "## Op Families",
            "",
        ]
    )
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(families.items()))
    lines.extend(
        [
            "",
            f"Total ATen ops: `{len(operations)}`",
            f"Tensor layout nodes: `{len(tensor_layouts)}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_event_artifacts(
    output_dir: Path,
    event: dict[str, str],
    event_ops: list[dict[str, str]],
) -> dict[str, Any]:
    event_dir = output_dir / slug(event["event_id"])
    event_dir.mkdir(parents=True, exist_ok=True)
    tensor_layouts, operations, edges = build_graph(event, event_ops)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "notes": event_role_note(event),
        "tensor_layouts": tensor_layouts,
        "operations": operations,
        "edges": [{"source": src, "target": dst, "label": label} for src, dst, label in edges],
        "lineage_note": (
            "Dispatch trace records tensor layouts but not stable tensor object IDs; "
            "edges reuse the most recent matching output layout when possible."
        ),
    }
    (event_dir / "tensor_layouts.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_example_py(event_dir / "example.py", event, tensor_layouts, operations)
    write_dot(event_dir / "tensor_graph.dot", event, tensor_layouts, operations, edges)
    write_svg(event_dir / "tensor_graph.svg", event, operations, tensor_layouts)
    write_summary_md(event_dir / "README.md", event, operations, tensor_layouts)
    return {
        "event_id": event["event_id"],
        "phase": event["phase"],
        "priority": event["priority"],
        "layer_id": event["layer_id"],
        "q_len": event["q_len"],
        "past_len": event["past_len"],
        "kv_len": event["kv_len"],
        "token_state": event["token_state"],
        "visipruner_role": event["visipruner_role"],
        "selection_result": event.get("selection_result", ""),
        "op_count": len(operations),
        "tensor_layout_count": len(tensor_layouts),
        "example_py": str((event_dir / "example.py").relative_to(output_dir)),
        "tensor_layouts_json": str((event_dir / "tensor_layouts.json").relative_to(output_dir)),
        "tensor_graph_dot": str((event_dir / "tensor_graph.dot").relative_to(output_dir)),
        "tensor_graph_svg": str((event_dir / "tensor_graph.svg").relative_to(output_dir)),
    }


def write_index(output_dir: Path, source_dir: Path, rows: list[dict[str, Any]]) -> None:
    phase_counts = Counter(row["phase"] for row in rows)
    role_counts = Counter(row["visipruner_role"] for row in rows)
    lines = [
        "# Dispatch Tensor-Layout Visualizations",
        "",
        "Generated from the VisiPrune-centric filtered TorchDispatch trace.",
        "",
        f"- source dispatch dir: `{source_dir}`",
        f"- strong-associated dispatch events: `{len(rows)}`",
        f"- phase counts: `{dict(sorted(phase_counts.items()))}`",
        "",
        "## Role Counts",
        "",
    ]
    lines.extend(f"- `{role}`: {count}" for role, count in sorted(role_counts.items()))
    lines.extend(
        [
            "",
            "## Event Artifacts",
            "",
            "| event | phase | role | q/past/kv | ops | example | graph |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        qpk = f"{row['q_len']}/{row['past_len']}/{row['kv_len']}"
        lines.append(
            f"| `{row['event_id']}` | `{row['phase']}` | `{row['visipruner_role']}` | "
            f"`{qpk}` | {row['op_count']} | "
            f"[example.py]({row['example_py']}) | [tensor_graph.svg]({row['tensor_graph_svg']}) |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `example.py` files use a `tensor_layouts` dictionary to describe the layer computation without requiring model weights.",
            "- `tensor_graph.dot` contains the shape-level directed graph source.",
            "- `tensor_graph.svg` is a standalone rendered op/layout graph for quick inspection.",
            "- Lineage is inferred from matching tensor layouts because dispatch metadata does not include stable tensor object IDs.",
            "",
        ]
    )
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-dir", default=str(DEFAULT_DISPATCH_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dispatch_dir = Path(args.dispatch_dir)
    output_dir = Path(args.output_dir)
    manifest_path = dispatch_dir / "dispatch_manifest.csv"
    ops_path = dispatch_dir / "dispatch_ops.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing dispatch manifest: {manifest_path}")
    if not ops_path.exists():
        raise SystemExit(f"Missing dispatch op trace: {ops_path}")

    manifest = read_csv(manifest_path)
    op_rows = read_csv(ops_path)
    ops_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in op_rows:
        ops_by_event[row["event_id"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for event in manifest:
        event_id = event["event_id"]
        if event_id not in ops_by_event:
            raise SystemExit(f"Manifest event has no dispatch ops: {event_id}")
        index_rows.append(write_event_artifacts(output_dir, event, ops_by_event[event_id]))

    write_csv(output_dir / "visualization_index.csv", index_rows)
    (output_dir / "visualization_index.json").write_text(
        json.dumps(index_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_index(output_dir, dispatch_dir, index_rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "event_count": len(index_rows),
                "dispatch_dir": str(dispatch_dir),
                "index": str(output_dir / "index.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
