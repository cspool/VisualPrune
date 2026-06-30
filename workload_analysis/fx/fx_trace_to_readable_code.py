#!/usr/bin/env python3
"""Render an FX dynamic trace JSON into readable Python-like code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_nodes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected node list in {path}")
    return payload


def expr(value: Any) -> str:
    if isinstance(value, dict):
        if "node" in value:
            return str(value["node"])
        if "tuple" in value:
            return "(" + ", ".join(expr(item) for item in value["tuple"]) + ("," if len(value["tuple"]) == 1 else "") + ")"
        if "list" in value:
            return "[" + ", ".join(expr(item) for item in value["list"]) + "]"
        if "dict" in value:
            items = [f"{key!r}: {expr(item)}" for key, item in value["dict"].items()]
            return "{" + ", ".join(items) + "}"
        if "slice" in value:
            item = value["slice"]
            return f"slice({expr(item.get('start'))}, {expr(item.get('stop'))}, {expr(item.get('step'))})"
        if "torch_dtype" in value:
            return f"torch.{value['torch_dtype']}"
        if "torch_device" in value:
            return f"torch.device({value['torch_device']!r})"
        if "tensor_literal" in value:
            meta = value["tensor_literal"]
            return f"<tensor shape={meta.get('shape')} dtype={meta.get('dtype')} device={meta.get('device')}>"
        if "repr" in value:
            return str(value["repr"])
    return repr(value)


def args_expr(args_payload: Any, kwargs_payload: Any) -> str:
    if isinstance(args_payload, dict) and "tuple" in args_payload:
        positional = [expr(item) for item in args_payload["tuple"]]
    else:
        positional = [expr(args_payload)]
    keyword_items: list[str] = []
    if isinstance(kwargs_payload, dict) and "dict" in kwargs_payload:
        keyword_items = [f"{key}={expr(value)}" for key, value in kwargs_payload["dict"].items()]
    return ", ".join(positional + keyword_items)


def output_expr(args_payload: Any) -> str:
    if isinstance(args_payload, dict) and "tuple" in args_payload:
        values = args_payload["tuple"]
        if len(values) == 1:
            return expr(values[0])
        return expr(args_payload)
    return expr(args_payload)


def node_comment(node: dict[str, Any]) -> str:
    meta = node.get("meta") or {}
    tensor_meta = meta.get("tensor_meta")
    if not isinstance(tensor_meta, dict):
        val = meta.get("val")
        if isinstance(val, dict):
            tensor_meta = val
    if not isinstance(tensor_meta, dict):
        return ""
    shape = tensor_meta.get("shape")
    dtype = tensor_meta.get("dtype")
    if shape is None and "attrs" in tensor_meta:
        attrs = tensor_meta.get("attrs") or {}
        shape = attrs.get("shape")
        dtype = attrs.get("dtype")
    parts = []
    if shape is not None:
        parts.append(f"shape={shape}")
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    return "  # " + ", ".join(parts) if parts else ""


def render_code(nodes: list[dict[str, Any]], function_name: str) -> str:
    placeholders = [node["name"] for node in nodes if node.get("op") == "placeholder"]
    lines = [
        "import operator",
        "import torch",
        "",
        "",
        f"def {function_name}({', '.join(placeholders)}):",
    ]
    if not nodes:
        lines.append("    pass")
        return "\n".join(lines) + "\n"

    emitted = False
    for node in nodes:
        op = node.get("op")
        name = node.get("name")
        target = str(node.get("target_expr") or node.get("target"))
        call_args = args_expr(node.get("args"), node.get("kwargs"))
        comment = node_comment(node)
        if op == "placeholder":
            lines.append(f"    # placeholder: {name}{comment}")
            continue
        if op == "get_attr":
            lines.append(f"    {name} = self.{node.get('target')}{comment}")
            emitted = True
            continue
        if op == "call_function":
            lines.append(f"    {name} = {target}({call_args}){comment}")
            emitted = True
            continue
        if op == "call_method":
            args_payload = node.get("args")
            if isinstance(args_payload, dict) and "tuple" in args_payload and args_payload["tuple"]:
                receiver = expr(args_payload["tuple"][0])
                rest = {"tuple": args_payload["tuple"][1:]}
                rest_args = args_expr(rest, node.get("kwargs"))
                lines.append(f"    {name} = {receiver}.{node.get('target')}({rest_args}){comment}")
            else:
                lines.append(f"    {name} = <call_method {node.get('target')}>({call_args}){comment}")
            emitted = True
            continue
        if op == "call_module":
            lines.append(f"    {name} = self.{node.get('target')}({call_args}){comment}")
            emitted = True
            continue
        if op == "output":
            lines.append(f"    return {output_expr(node.get('args'))}")
            emitted = True
            continue
        lines.append(f"    # unsupported FX op {op}: {name} = {target}({call_args})")
    if not emitted:
        lines.append("    pass")
    return "\n".join(lines) + "\n"


def render_markdown(nodes: list[dict[str, Any]], code_path: Path) -> str:
    lines = [
        "# FX Readable Process",
        "",
        f"Generated code: `{code_path.name}`",
        "",
        "| index | name | op | target | users |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for node in nodes:
        users = ", ".join(node.get("users") or [])
        lines.append(
            f"| {node.get('index')} | `{node.get('name')}` | `{node.get('op')}` | "
            f"`{node.get('target')}` | `{users}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", help="Directory containing fx_nodes.json.")
    parser.add_argument("--nodes-json", help="Path to fx_nodes.json. Overrides --trace-dir.")
    parser.add_argument("--recursive", action="store_true", help="Process every fx_nodes.json under --trace-dir.")
    parser.add_argument("--output", help="Output Python file. Defaults to <trace-dir>/fx_readable_process.py.")
    parser.add_argument("--markdown", help="Output Markdown index. Defaults to <trace-dir>/fx_readable_process.md.")
    parser.add_argument("--function-name", default="fx_readable_process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nodes_json and args.recursive:
        raise ValueError("--recursive can only be used with --trace-dir")
    if args.recursive:
        if not args.trace_dir:
            raise ValueError("--recursive requires --trace-dir")
        root = Path(args.trace_dir)
        node_paths = sorted(root.rglob("fx_nodes.json"))
        if not node_paths:
            raise FileNotFoundError(f"No fx_nodes.json files found under {root}")
        results = []
        for nodes_path in node_paths:
            base_dir = nodes_path.parent
            nodes = load_nodes(nodes_path)
            code_path = base_dir / "fx_readable_process.py"
            markdown_path = base_dir / "fx_readable_process.md"
            code_path.write_text(render_code(nodes, args.function_name), encoding="utf-8")
            markdown_path.write_text(render_markdown(nodes, code_path), encoding="utf-8")
            results.append({
                "nodes": str(nodes_path),
                "code": str(code_path),
                "markdown": str(markdown_path),
                "node_count": len(nodes),
            })
        print(json.dumps({"processed": len(results), "results": results}, indent=2, ensure_ascii=False))
        return

    if args.nodes_json:
        nodes_path = Path(args.nodes_json)
        base_dir = nodes_path.parent
    elif args.trace_dir:
        base_dir = Path(args.trace_dir)
        nodes_path = base_dir / "fx_nodes.json"
    else:
        raise ValueError("Provide --trace-dir or --nodes-json")

    nodes = load_nodes(nodes_path)
    code_path = Path(args.output) if args.output else base_dir / "fx_readable_process.py"
    markdown_path = Path(args.markdown) if args.markdown else base_dir / "fx_readable_process.md"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(render_code(nodes, args.function_name), encoding="utf-8")
    markdown_path.write_text(render_markdown(nodes, code_path), encoding="utf-8")
    print(json.dumps({
        "nodes": str(nodes_path),
        "code": str(code_path),
        "markdown": str(markdown_path),
        "node_count": len(nodes),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
