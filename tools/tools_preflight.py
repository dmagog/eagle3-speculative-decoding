# -*- coding: utf-8 -*-
"""Предполётная проверка перед `kaggle kernels push`.

Единственная задача: пуш НИКОГДА не должен снимать ноутбук с публикации.
Перевод в Public — осознанное действие автора (§12 гайда), и метаданные в репозитории
легко остаются со старым `is_private: true` после того, как автор опубликовал руками.
Именно так и вышло: ноутбук опубликован, а локально лежало `true`.

Запуск:  python3 tools_preflight.py [notebook|notebook_en]
Код 1 — пушить нельзя.
"""
import json
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
# каталог ядра передаётся аргументом: ядра теперь два, и оба публичные
DIR = sys.argv[1] if len(sys.argv) > 1 else "notebook"
# путь ищем и от самого инструмента, и от корня репозитория, и от текущего каталога:
# в рабочей папке это notebook/, в репозитории — kernel/ru
META = next((p for p in (os.path.join(HERE, DIR, "kernel-metadata.json"),
                         os.path.join(os.path.dirname(HERE), DIR, "kernel-metadata.json"),
                         os.path.join(DIR, "kernel-metadata.json"))
             if os.path.exists(p)), os.path.join(HERE, DIR, "kernel-metadata.json"))


def live_metadata(ref):
    """Через curl, а не urllib: у системного Python на этом маке нет корневых
    сертификатов, и urlopen падает на CERTIFICATE_VERIFY_FAILED."""
    cfg = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json"), encoding="utf-8"))
    user, slug = ref.split("/", 1)
    url = ("https://www.kaggle.com/api/v1/kernels/pull"
           f"?user_name={user}&kernel_slug={slug}")
    out = subprocess.run(
        ["curl", "-sS", "--fail", "--show-error", "-u", f"{cfg['username']}:{cfg['key']}", url],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        err = out.stderr.strip()
        if "404" in err:                        # ядра ещё нет
            raise FileNotFoundError(err[:120])
        raise RuntimeError(err[:120])
    return json.loads(out.stdout).get("metadata", {})


def main():
    local = json.load(open(META, encoding="utf-8"))
    ref = local["id"]
    try:
        live = live_metadata(ref)
    except FileNotFoundError:                   # ядра ещё нет — первая публикация
        print("ноутбук на Kaggle не найден: считаю это первой публикацией, пушить можно")
        return 0
    except Exception as e:
        print(f"СТОП: живое состояние не проверено ({e}).")
        print("Пуш заблокирован: пока не доказано, что он не снимет ноутбук с публикации.")
        return 1

    # поля нет — считаем приватным: осторожная сторона
    raw = local.get("is_private", "true")
    local_private = str(raw).lower() in ("true", "1")
    lp = live.get("isPrivate")
    live_private = str(lp).lower() in ("true", "1")   # API отвечает то bool, то строкой

    print(f"локально is_private={local_private} | на Kaggle isPrivate={live_private}")
    if not live_private and local_private:
        print("\nСТОП: ноутбук ОПУБЛИКОВАН, а локальные метаданные снимут его с публикации.")
        print(f"Почини: is_private -> \"false\" в {META}")
        return 1
    if live_private and not local_private:
        print("! внимание: ноутбук на Kaggle приватный, а пуш его ОПУБЛИКУЕТ.")

    bad = [k for k in local.get("keywords", []) if k in ("llm", "beginner", "deep learning")]
    if bad:
        print(f"! теги, которые Kaggle отвергает и молча выбрасывает: {bad}")
    print("предполётная проверка пройдена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
