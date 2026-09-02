# -*- coding: utf-8 -*-
"""Статическая проверка ноутбука: имена, используемые до определения.
Ловит класс ошибок, который переживает и preflight, и вычитку, и локальный рендер
фигуры, — но валит прогон на Kaggle (так ушли BEST_SIZE и eaglegenerate).
Usage: python3 tools_cellcheck.py <path to .ipynb>"""
import ast
import builtins
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
# имена, которые Jupyter/Kaggle дают сами
defined = set(dir(builtins)) | {"display", "get_ipython", "In", "Out", "__file__"}
problems = []


class Collect(ast.NodeVisitor):
    """Всё, что ячейка вводит в пространство имён."""

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Store):
            defined.add(n.id)

    def visit_FunctionDef(self, n):
        defined.add(n.name)
        for a in list(n.args.args) + list(n.args.kwonlyargs) + list(n.args.posonlyargs):
            defined.add(a.arg)
        for a in (n.args.vararg, n.args.kwarg):
            if a:
                defined.add(a.arg)
        self.generic_visit(n)

    def visit_Lambda(self, n):
        for a in list(n.args.args) + list(n.args.kwonlyargs):
            defined.add(a.arg)
        for a in (n.args.vararg, n.args.kwarg):
            if a:
                defined.add(a.arg)
        self.generic_visit(n)

    def visit_ClassDef(self, n):
        defined.add(n.name)
        self.generic_visit(n)

    def visit_Import(self, n):
        for a in n.names:
            defined.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            defined.add(a.asname or a.name)

    def _bind(self, node):
        for t in ast.walk(node):
            if isinstance(t, ast.Name):
                defined.add(t.id)

    def visit_For(self, n):
        self._bind(n.target)
        self.generic_visit(n)

    def visit_comprehension(self, n):
        self._bind(n.target)
        self.generic_visit(n)

    def visit_ExceptHandler(self, n):
        if n.name:
            defined.add(n.name)
        self.generic_visit(n)

    def visit_withitem(self, n):
        if n.optional_vars:
            self._bind(n.optional_vars)
        self.generic_visit(n)

    def visit_Global(self, n):
        for name in n.names:
            defined.add(name)


for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        problems.append((i, "SyntaxError", str(e)))
        continue
    Collect().visit(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined:
            problems.append((i, "используется до определения", n.id))

seen, out = set(), []
for i, kind, what in problems:
    if (kind, what) in seen:
        continue
    seen.add((kind, what))
    out.append((i, kind, what))

for i, kind, what in out:
    print(f"  ячейка {i}: {kind}: {what}")
print(f"\nподозрительных: {len(out)}")
sys.exit(1 if out else 0)
