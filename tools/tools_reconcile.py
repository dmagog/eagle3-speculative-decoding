# -*- coding: utf-8 -*-
"""Сверяет зашитые в прозу числа с key_numbers.json прогона.
Запуск: python3 reconcile.py <путь к key_numbers.json>"""
import json
import re
import sys

import os
def _find(name, *dirs):
    """Один и тот же файл лежит по-разному в рабочей папке и в репозитории;
    ищем по обоим раскладам, чтобы копии инструмента не разъезжались."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for base in (here, root):
        for d in dirs:
            p = os.path.join(base, d, name)
            if os.path.exists(p):
                return p
    raise SystemExit("не нашёл " + name)


NB = _find("eagle3-qwen3.ipynb", "notebook", os.path.join("notebooks", "ru"))

k = json.load(open(sys.argv[1], encoding="utf-8"))
nb = json.load(open(NB, encoding="utf-8"))
# билдер пишет source строкой, а выгрузка из Kaggle — списком строк
_src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
md = " ".join(" ".join(_src(c).split())
               for c in nb["cells"] if c["cell_type"] == "markdown")

eng, rus, ov = k["english"], k["russian"], k["overall"]
by = k["by_set"]
math_code = [by[b] for b in by if b in ("GSM8K", "HumanEval")]
mc = sum(x["speedup"] for x in math_code) / len(math_code) if math_code else float("nan")

print("=== что показал прогон ===")
print(f"  всего      τ {ov['tau']:.2f} | {ov['speedup_naive']:.2f}x к naive | {ov['speedup_hf']:.2f}x к generate")
print(f"  английские τ {eng['tau']:.2f} | {eng['speedup']:.2f}x")
print(f"  русский    τ {rus['tau']:.2f} | {rus['speedup']:.2f}x")
for b, v in by.items():
    print(f"    {b:11s} τ {v['tau']:.2f} | {v['speedup']:.2f}x")
print(f"  математика+код (GSM8K, HumanEval): {mc:.2f}x")
print(f"  ветвление: +{k['branching']['gain_pct']:.0f}% (дерево {k['branching']['tau_tree']:.2f} против цепочки {k['branching']['tau_chain']:.2f})")
t = k["temperature"]
print("  температура:", ", ".join(f"T={a}: {b['speedup']:.2f}x" for a, b in t.items()))

# --- утверждения, зашитые в текст: (что ищем, ожидаемое значение, допуск) ---
CLAIMS = [
    # русское ускорение гуляет 0.93-0.95 между прогонами — допуск по факту разброса
    ("на русских запросах — около 0.95", 0.95, rus["speedup"], 0.03),
    ("2.3× on the English benchmarks", 2.3, eng["speedup"], 0.06),
    ("acceptance length τ ≈ 3.4", 3.4, eng["tau"], 0.12),
    ("длина принятия держится около 3.4", 3.4, eng["tau"], 0.12),
    ("Branching adds 39% to", 39.0, k["branching"]["gain_pct"], 4.0),
    ("на русских она падает примерно до 1.4", 1.4, rus["tau"], 0.12),
    ("около двух с половиной раз (математика и код)", 2.5, mc, 0.15),
    ("≈0.95× — an actual", 0.95, rus["speedup"], 0.03),
]

print("\n=== сверка ===")
bad = 0
for phrase, claimed, actual, tol in CLAIMS:
    key = phrase.split(" (")[0]
    present = key in md
    ok = abs(claimed - actual) <= tol
    mark = "ок " if ok else "РАСХОЖДЕНИЕ"
    note = "" if present else "   [фразы в тексте нет — проверить вручную]"
    if not ok:
        bad += 1
    print(f"  {mark:12s} заявлено {claimed:6.2f} | замер {actual:6.2f} | {phrase}{note}")
# --- §11, замер масштабирования ---------------------------------------------------
# Раздел спорит о разнице в проценты, а его абсолютные числа зависят от формы дерева,
# которую §7 выбирает на каждом прогоне заново. Поэтому проверяем то, что пережило
# смену дерева в английской версии: относительный прирост τ и порядок величин.
sc = k.get("scaling")
if not sc:
    print("\n!! в key_numbers.json нет блока 'scaling' — текст §11 сверить нечем")
    bad += 1
else:
    a, b = sc["m17"], sc["m4b"]
    d_tau = 100 * (b["in"]["tau"] / a["in"]["tau"] - 1)
    d_tau_out = 100 * (b["out"]["tau"] / a["out"]["tau"] - 1)
    d_sp = 100 * (b["in"]["speedup"] / a["in"]["speedup"] - 1)
    cyc = [a["in"]["eagle_ms"] * a["in"]["tau"], b["in"]["eagle_ms"] * b["in"]["tau"]]
    step = [a["in"]["naive_ms"], b["in"]["naive_ms"]]
    pct = lambda v: 100 * (v[1] / v[0] - 1)
    print(f"\n=== §11 масштаб ({a['in'].get('reps','?')} прохода на точку, дерево {sc['tree']}) ===")
    print(f"  шаг   {step[0]:5.1f} -> {step[1]:5.1f} мс ({pct(step):+5.1f}%)")
    print(f"  цикл  {cyc[0]:5.1f} -> {cyc[1]:5.1f} мс ({pct(cyc):+5.1f}%)")
    print(f"  τ      {a['in']['tau']:4.2f} ->  {b['in']['tau']:4.2f}    ({d_tau:+5.1f}%)  "
          f"| вне домена {a['out']['tau']:4.2f} -> {b['out']['tau']:4.2f} ({d_tau_out:+5.1f}%)")
    print(f"  ускорение в домене {a['in']['speedup']:.2f}x -> {b['in']['speedup']:.2f}x ({d_sp:+.1f}%)"
          f" | вне домена {a['out']['speedup']:.2f}x -> {b['out']['speedup']:.2f}x")
    for label, claimed, actual, tol in (
        ("около 9% в пользу пары 4B (в домене)", 9, d_tau, 2.0),
        ("около 9% и вне домена", 9, d_tau_out, 3.0),
        ("длина принятия против §5 почти не сдвинулась", k["russian"]["tau"], a["out"]["tau"], 0.06),
    ):
        ok = abs(claimed - actual) <= tol
        if not ok:
            bad += 1
        print(f"  {'ок ' if ok else 'РАСХОЖДЕНИЕ':12s} заявлено {claimed:6.2f} | замер {actual:6.2f} | {label}")
    print("\n=== §11 утверждения ===")
    for label, cond in (
        ("цикл дорожает быстрее обычного шага", pct(cyc) > pct(step)),
        ("длина принятия растёт вместе с целью", b["in"]["tau"] > a["in"]["tau"]),
        ("разница в ускорении укладывается в пару процентов", abs(d_sp) < 5.0),
        ("вне домена обе пары выше 1.0", a["out"]["speedup"] > 1.0 and b["out"]["speedup"] > 1.0),
        ("жадное декодирование даёт нулевой разброс τ",
         max(a["in"]["tau_err"], b["in"]["tau_err"], a["out"]["tau_err"], b["out"]["tau_err"]) < 5e-4),
    ):
        if not cond:
            bad += 1
        print(f"  {'ок ' if cond else 'УТВЕРЖДЕНИЕ СЛОМАНО':12s} {label}")

# --- §7: срез по глубине на чужом домене (то, чем §5 объясняет своё замедление) -----
ab = k.get("ablation")
if not ab or "depth_ru" not in ab:
    print("\n!! в key_numbers.json нет 'ablation.depth_ru' — утверждение §5/§7 про глубину не проверено")
    bad += 1
else:
    nru = ab["naive_ms_ru"]
    ru = {int(kk): v for kk, v in ab["depth_ru"].items()}
    ds = sorted(ru)
    sp = {d: nru / ru[d]["ms"] for d in ds}
    taus = [ru[d]["tau"] for d in ds]
    print("\n=== §7 глубина на русском наборе ===")
    print("  " + " | ".join(f"d{d}: {sp[d]:.2f}x (τ {ru[d]['tau']:.2f})" for d in ds))
    for label, cond in (
        ("на мелкой глубине выигрыш (> 1.0)", sp[ds[0]] > 1.0),
        ("на глубоком дереве проигрыш (< 1.0)", sp[ds[-1]] < 1.0),
        ("кривая монотонно падает с глубиной",
         all(sp[ds[i]] >= sp[ds[i + 1]] - 0.02 for i in range(len(ds) - 1))),
        ("длина принятия почти не зависит от глубины (разброс < 0.15)",
         max(taus) - min(taus) < 0.15),
    ):
        if not cond:
            bad += 1
        print(f"  {'ок ' if cond else 'УТВЕРЖДЕНИЕ СЛОМАНО':12s} {label}")

# --- §6: доли принятия, названные в тексте словами -------------------------------
ac = k.get("acceptance")
if not ac:
    print("\n!! в key_numbers.json нет 'acceptance' — доли §6 не проверены")
    bad += 1
else:
    al = ac["alpha"]
    print(f"\n=== §6 доли принятия ({ac['n_cycles']} циклов) ===")
    print("  " + " | ".join(f"a>={d+1}: {v:.2f}" for d, v in enumerate(al)))
    for label, claimed, actual, tol in (
        ("хотя бы один черновик принимают 72% циклов", 0.72, al[0], 0.06),
        ("до второго уровня доходят 35%", 0.35, al[1] if len(al) > 1 else -1, 0.06),
        ("до четвёртого — 7%", 0.07, al[3] if len(al) > 3 else -1, 0.05),
    ):
        ok = abs(claimed - actual) <= tol
        if not ok:
            bad += 1
        print(f"  {'ок ' if ok else 'РАСХОЖДЕНИЕ':12s} заявлено {claimed:5.2f} | замер {actual:5.2f} | {label}")
    deepest = ac["max_accept_length"] == 4
    if not deepest:
        bad += 1
    print(f"  {'ок ' if deepest else 'УТВЕРЖДЕНИЕ СЛОМАНО':12s} четвёртый уровень и правда предел "
          f"(замер {ac['max_accept_length']})")

print(f"\nрасхождений: {bad}")
