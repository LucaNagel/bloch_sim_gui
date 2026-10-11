"""Evaluate mathematical parameter-loop formulas on an evenly spaced grid."""

from __future__ import annotations

import ast
import operator
import re

import numpy as np


_FUNCTIONS = {
    name: getattr(np, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "log10",
        "log2",
        "sqrt",
        "abs",
        "minimum",
        "maximum",
        "clip",
    )
}
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def parameter_loop_values(formula: str, start: float, stop: float, count: int):
    """Return ``f(linspace(start, stop, count))`` without executing Python code.

    Accept ``^`` and ``**`` for powers, an optional ``y =`` prefix, and common
    NumPy mathematical functions with or without the ``np.`` prefix. Scalar
    formulas produce the same parameter value at every point.
    """
    expression = re.sub(r"^y\s*=\s*", "", str(formula).strip()).replace("^", "**")
    if not expression:
        raise ValueError("Enter a formula, for example x, x^2, or sin(x).")
    if len(expression) > 1000:
        raise ValueError("The formula is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid formula syntax. Try x^2 or sin(x).") from exc
    if sum(1 for _ in ast.walk(tree)) > 200:
        raise ValueError("The formula is too complex.")
    if count < 1 or not np.isfinite([start, stop]).all():
        raise ValueError("Use finite endpoints and at least one point.")
    x = np.linspace(start, stop, count, dtype=float)
    names = {"x": x, "pi": np.pi, "e": np.e}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return np.float64(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("np", "numpy") and node.attr in ("pi", "e"):
                return names[node.attr]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and not node.keywords:
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id in ("np", "numpy")
            ):
                function = ast.Name(id=function.attr)
            if isinstance(function, ast.Name) and function.id in _FUNCTIONS:
                count = {"minimum": 2, "maximum": 2, "clip": 3}.get(function.id, 1)
                if len(node.args) != count:
                    raise ValueError(f"{function.id} requires {count} argument(s).")
                return _FUNCTIONS[function.id](*(evaluate(arg) for arg in node.args))
        raise ValueError(
            "Use only x, pi, e, arithmetic, and mathematical functions "
            "such as sin, cos, exp, log, sqrt, or abs."
        )

    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            values = np.asarray(evaluate(tree.body), dtype=float)
    except (FloatingPointError, OverflowError, TypeError) as exc:
        raise ValueError(
            "The formula must produce finite real values throughout the x range."
        ) from exc
    if values.ndim == 0:
        values = np.full(x.shape, float(values))
    if values.shape != x.shape or not np.isfinite(values).all():
        raise ValueError("The formula must produce one finite real value per x point.")
    return values
