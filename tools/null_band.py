"""
Пять способов построить нулевую полосу для сравнения двух эмпирических
распределений — и доля ложных срабатываний у каждого.

Обе выборки берутся из ОДНОГО распределения, то есть правильный ответ всегда
«различий нет». Честный тест при alpha = 5% обязан ошибаться примерно в 5%
прогонов. Скрипт показывает, что четыре способа из пяти этого не делают,
причём в обе стороны.

python3 null_band.py
"""
import numpy as np

K, N = 80, 500          # носитель распределения и размер каждой выборки
TRIALS, REPS = 1000, 500
ALPHA = 0.05
SEED = 20260916


def tv(a, b):
    """Расстояние полной вариации между двумя векторами счётчиков."""
    return 0.5 * np.abs(a / a.sum() - b / b.sum()).sum()


def run():
    rng = np.random.default_rng(SEED)
    w = 1.0 / np.arange(1, K + 1) ** 1.1      # разреженный носитель, тяжёлая голова
    P = w / w.sum()

    names = [
        "перестановочный (правильный)",
        "бутстрэп из объединённого пула",
        "бутстрэп: каждая из своей выборки",
        "бутстрэп из одной выборки",
        "ресэмпл против исходной выборки",
    ]
    rejected = np.zeros(len(names))
    thresholds = [[] for _ in names]
    observed = []

    for _ in range(TRIALS):
        a = rng.multinomial(N, P)             # обе выборки из одного распределения
        b = rng.multinomial(N, P)
        obs = tv(a, b)
        observed.append(obs)

        pooled = a + b
        p_pool, p_a, p_b = pooled / pooled.sum(), a / N, b / N

        # 1. Перестановочный: объединённый пул делится пополам без возвращения.
        #    Если распределения совпадают, метки обменимы — это точная нулевая статистика.
        splits = rng.multivariate_hypergeometric(pooled, N, size=REPS)
        nulls = [np.array([tv(s, pooled - s) for s in splits])]

        # 2-4. Бутстрэп: из каких распределений тянуть две искусственные выборки.
        for src_a, src_b in ((p_pool, p_pool), (p_a, p_b), (p_a, p_a)):
            x = rng.multinomial(N, src_a, size=REPS)
            y = rng.multinomial(N, src_b, size=REPS)
            nulls.append(np.array([tv(u, v) for u, v in zip(x, y)]))

        # 5. Ресэмпл сравнивается с исходной выборкой, а не со вторым ресэмплом.
        #    Полоса меряет шум ОДНОЙ выборки, а наблюдаемое TV — расстояние между ДВУМЯ.
        x = rng.multinomial(N, p_a, size=REPS)
        nulls.append(np.array([tv(u, a) for u in x]))

        for i, null in enumerate(nulls):
            thresholds[i].append(np.percentile(null, 95))
            if (null >= obs).mean() <= ALPHA:
                rejected[i] += 1

    reference = np.median(thresholds[0])
    print(f"{TRIALS} прогонов, обе выборки из одного распределения, "
          f"n={N} на выборку, alpha={ALPHA:.0%}")
    print(f"типичное наблюдаемое TV = {np.median(observed):.4f}\n")
    print(f"{'как построена нулевая полоса':<36}{'порог':>8}{'к эталону':>11}{'ложных':>9}")
    for i, name in enumerate(names):
        median = np.median(thresholds[i])
        print(f"{name:<36}{median:8.4f}{100 * median / reference:10.0f}%"
              f"{100 * rejected[i] / TRIALS:8.1f}%")


if __name__ == "__main__":
    run()
