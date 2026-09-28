"""
Рисунок к статье: ложное срабатывание — это всё, что правее порога.

Одна и та же гистограмма в каждой панели: расстояния между парами выборок
из ОДНОГО распределения, тысяча прогонов. Меняется только порог — медиана
95-го процентиля нулевого распределения, построенного данным способом.
Закрашенный хвост за порогом и есть ложные срабатывания.

Данные — та же симуляция и то же зерно, что в tools/null_band.py.
python3 draw_null_band.py
"""
import numpy as np
import matplotlib.pyplot as plt

K, N, TRIALS, REPS, SEED = 80, 500, 1000, 500, 20260916
HABR_BODY_PX = 730                      # как у рисунков первой статьи; сохраняем в 2x

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
BASE, GRID = "#c3c2b7", "#e1e0d9"
FILL, TAIL = "#8e8c85", "#eb6834"      # проверено validate_palette.js: CVD 10.3, NV 17.2, контраст ≥ 3:1


def simulate():
    rng = np.random.default_rng(SEED)
    w = 1.0 / np.arange(1, K + 1) ** 1.1
    P = w / w.sum()

    def tv(a, b):
        return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()

    obs_all, thr, rej = [], [[] for _ in range(5)], np.zeros(5)
    for _ in range(TRIALS):
        a, b = rng.multinomial(N, P), rng.multinomial(N, P)
        obs = tv(a, b)
        obs_all.append(obs)
        pooled = a + b
        pp, pa, pb = pooled / pooled.sum(), a / N, b / N
        splits = rng.multivariate_hypergeometric(pooled, N, size=REPS)
        nulls = [np.array([tv(s, pooled - s) for s in splits])]
        for sa, sb in ((pp, pp), (pa, pb), (pa, pa)):
            x, y = rng.multinomial(N, sa, size=REPS), rng.multinomial(N, sb, size=REPS)
            nulls.append(np.array([tv(u, v) for u, v in zip(x, y)]))
        x = rng.multinomial(N, pa, size=REPS)
        nulls.append(np.array([tv(u, a) for u in x]))
        for i, null in enumerate(nulls):
            thr[i].append(np.percentile(null, 95))
            if (null >= obs).mean() <= 0.05:
                rej[i] += 1
    return np.array(obs_all), np.median(np.array(thr), axis=1), 100 * rej / TRIALS


def draw(obs, thresholds, fpr, path):
    names = [                               # порядок — как строки таблицы в статье
        "Перестановочный способ (эталон)",
        "Бутстрэп из объединённого пула",
        "Бутстрэп из каждой выборки отдельно",
        "Бутстрэп из одной выборки",
        "Ресэмпл против исходной выборки",
    ]
    edges = np.arange(0.10, 0.301, 0.004)
    counts, _ = np.histogram(obs, bins=edges)
    width = edges[1] - edges[0]

    fig, axes = plt.subplots(5, 1, figsize=(8.0, 6.4), sharex=True)
    fig.subplots_adjust(left=0.385, right=0.965, top=0.875, bottom=0.10, hspace=0.35)

    for ax, name, t, f in zip(axes, names, thresholds, fpr):
        for left, c in zip(edges[:-1], counts):
            right = left + width
            if right <= t:                                   # целиком левее порога
                ax.bar(left, c, width=width, align="edge", color=FILL, linewidth=0)
            elif left >= t:                                  # целиком в хвосте
                ax.bar(left, c, width=width, align="edge", color=TAIL, linewidth=0)
            else:                                            # столбец режется порогом
                ax.bar(left, c, width=t - left, align="edge", color=FILL, linewidth=0)
                ax.bar(t, c, width=right - t, align="edge", color=TAIL, linewidth=0)
        ax.axvline(t, color=INK, linewidth=1.6)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
        ax.set_xlim(0.10, 0.30)
        ax.set_xticks(np.arange(0.10, 0.301, 0.05))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.set_ylim(0, counts.max() * 1.08)
        # подпись строки: доля ложных — крупно, название способа — под ней
        ax.text(-0.03, 0.62, f"{f:.1f}%".replace(".0%", "%"), transform=ax.transAxes,
                ha="right", va="center", fontsize=15, fontweight="bold", color=INK)
        ax.text(-0.03, 0.18, name, transform=ax.transAxes,
                ha="right", va="center", fontsize=9, color=INK_2)

    fig.text(0.02, 0.965, "Ложное срабатывание — всё, что правее порога",
             ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, "Тысяча пар выборок без различий. Гистограмма во всех строках одна и та же, "
             "меняется только порог.", ha="left", va="top", fontsize=9, color=INK_2)
    axes[-1].set_xlabel("расстояние полной вариации между выборками", fontsize=9.5, color=INK_2)

    dpi = 2 * HABR_BODY_PX / fig.get_size_inches()[0]
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    obs, thresholds, fpr = simulate()
    for t, f in zip(thresholds, fpr):
        print(f"порог {t:.4f}  ложных {f:.1f}%")
    draw(obs, thresholds, fpr, "null_band_tails.png")
    print("сохранено: null_band_tails.png")
