#!/bin/zsh
# =============================================================================
# EAGLE-3 notebook -> PDF с отрендеренными формулами (offline, без сети)
# Отличия от предложенного варианта: изоляция npm, снос ВСЕХ ссылок на cdnjs
# (а не только MathJax), жёсткая сверка числа формул, корректный Title в PDF.
# Запуск:  tools/make_pdf.sh [корень репозитория]
# По умолчанию корень вычисляется от самого скрипта, а PDF ложатся рядом
# со своими .ipynb — в notebooks/ru и notebooks/en.
# =============================================================================
set -euo pipefail
REPO="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${TMPDIR:-/tmp}/nb2pdf-$$"
NFORMULAS=18                       # число формул в ноутбуке, сверяется жёстко
mkdir -p "$WORK"
cd "$WORK"

# --- 0. Установка KaTeX В ИЗОЛЯЦИИ -------------------------------------------
# Без собственного package.json npm уходит вверх по дереву каталогов, находит
# чужой package.json и ставит пакет ТУДА; $WORK/node_modules не появляется.
print '{"name":"nb2pdf","private":true}' > package.json
npm install --no-audit --no-fund --prefix "$WORK" katex@0.16.22
KATEX_DIST="$WORK/node_modules/katex/dist"
[[ -f "$KATEX_DIST/katex.min.js" ]] || { print -u2 "KaTeX не установился в $KATEX_DIST"; exit 1; }
# Если playwright отсутствует:
#   python3 -m pip install 'nbconvert[webpdf]' playwright && python3 -m playwright install chromium

# --- 1. _kg_hide-input -> tag remove_input -----------------------------------
cat > prep.py <<'PREP_EOF'
import json, sys
nb = json.load(open(sys.argv[1]))
n = 0
for c in nb['cells']:
    md = c.setdefault('metadata', {})
    if md.get('_kg_hide-input'):
        tags = md.setdefault('tags', [])
        if 'remove_input' not in tags:
            tags.append('remove_input')
        n += 1
json.dump(nb, open(sys.argv[2], 'w'))
print('tagged', n, file=sys.stderr)
PREP_EOF

# --- 2. CDN-MathJax -> встроенный KaTeX + снос ОСТАЛЬНЫХ ссылок на cdnjs -----
cat > katexify.py <<'KATEXIFY_EOF'
#!/usr/bin/env python3
"""nbconvert HTML -> полностью автономный HTML с формулами, отрисованными KaTeX.
Выбрасывает ВСЕ обращения к cdnjs, а не только MathJax: nbconvert тянет ещё
require.js (parser-blocking тег в <head>) и mermaid. Пока require.js оставался,
сборка ходила в сеть на каждом прогоне и зависала при медленном CDN."""
import base64, os, re, sys

html_in, html_out, katex_dist = sys.argv[1], sys.argv[2], sys.argv[3]

def read(p, b=False):
    return open(p, 'rb').read() if b else open(p, encoding='utf-8').read()

css = read(os.path.join(katex_dist, 'katex.min.css'))
css = re.sub(r'url\(fonts/([A-Za-z0-9_.-]+)\.woff2\)',
             lambda m: 'url(data:font/woff2;base64,%s)' % base64.b64encode(
                 read(os.path.join(katex_dist, 'fonts', m.group(1) + '.woff2'), True)).decode(),
             css)
css = re.sub(r',\s*url\(fonts/[A-Za-z0-9_.-]+\.(?:woff|ttf)\)\s*format\("(?:woff|truetype)"\)', '', css)

katex_js = read(os.path.join(katex_dist, 'katex.min.js'))
auto_js  = read(os.path.join(katex_dist, 'contrib', 'auto-render.min.js'))

BUNDLE = """<!-- Local KaTeX (offline, no CDN) -->
<style>%s
/* --- печать: правки под дефекты, найденные аудитом PDF ------------------ */
@media print {
  /* Без этого длинные строки кода срезаются границей блока — без многоточия
     и без полосы прокрутки, то есть текст просто пропадает. */
  pre, pre code, .highlight pre, .jp-OutputArea-output pre {
    white-space: pre-wrap;
    overflow-wrap: break-word;
    overflow-x: visible;
  }
  /* Заголовок не остаётся один внизу страницы. */
  h1, h2, h3, h4 { break-after: avoid-page; page-break-after: avoid; }
  /* Рамка врезки не разрывается между страницами. */
  div[style*="border-radius:6px"] { break-inside: avoid; page-break-inside: avoid; }
  /* Высокая фигура ужимается по месту вместо переноса на отдельную страницу;
     фигуры ниже порога не затрагиваются. */
  img { max-height: 88vh; width: auto; height: auto; }
}
</style>
<script>
/* Если require.js всё-таки просочится, его AMD define() перехватит UMD-обёртку
   KaTeX и глобалы не появятся. Прячем загрузчик на время этих двух скриптов. */
window.__amd_define = window.define; window.define = undefined;
window.__amd_module = window.module; window.module = undefined;
window.__amd_exports = window.exports; window.exports = undefined;
</script>
<script>%s</script>
<script>%s</script>
<script>
window.define = window.__amd_define; window.module = window.__amd_module;
window.exports = window.__amd_exports;
</script>
<script>
(function () {
  function mark(state, n, bad) {
    document.documentElement.setAttribute('data-math-count', String(n));
    document.documentElement.setAttribute('data-math-errors', String(bad));
    document.documentElement.setAttribute('data-math-ready', state);
  }
  function go() {
    try {
      renderMathInElement(document.body, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '\\\\[', right: '\\\\]', display: true},
          {left: '$',  right: '$',  display: false},
          {left: '\\\\(', right: '\\\\)', display: false}
        ],
        ignoredTags: ['script','noscript','style','textarea','pre','code','option'],
        throwOnError: false,
        strict: false
      });
    } catch (e) {
      mark('error:' + e.message, 0, 0);
      return;
    }
    var n   = document.querySelectorAll('.katex').length;
    var bad = document.querySelectorAll('.katex-error').length;
    // Готовность объявляем только когда веб-шрифты реально применимы, иначе
    // Chrome успеет напечатать кадр с запасными метриками.
    var fin = function () { mark(n > 0 ? 'ok' : 'empty', n, bad); };
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(fin, fin);
    } else { fin(); }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', go);
  } else { go(); }
})();
</script>
<!-- End of local KaTeX -->""" % (css, katex_js, auto_js)

h = read(html_in)

# 2.1 блок MathJax -> бандл KaTeX
start = h.index('<!-- Load mathjax -->')
end   = h.index('<!-- End of mathjax configuration -->') + len('<!-- End of mathjax configuration -->')
h = h[:start] + BUNDLE + h[end:]

# 2.2 убрать require.js с cdnjs (в этих ноутбуках он не используется:
#     ipywidgets-выводов нет, define(/require( встречаются только внутри
#     UMD-обёртки самого KaTeX)
h, n_req = re.subn(
    r'<script\s+src="https://cdnjs\.cloudflare\.com/ajax/libs/require\.js/[^"]*"\s*>\s*</script>',
    '', h)

# 2.3 обезвредить динамический import mermaid с cdnjs
h, n_mer = re.subn(r'https://cdnjs\.cloudflare\.com/ajax/libs/mermaid/[^"]*', 'about:blank', h)

# 2.4 гарантия: снаружи файла ничего не грузится
leftover = sorted(set(re.findall(r'(?:src|href)="(https?://[^"]*)"', h)
                      + re.findall(r'import\("(https?://[^"]*)"\)', h)))
ext = [u for u in leftover if u.endswith('.js') or '.min.' in u]
assert not ext, 'внешний подресурс уцелел: %r' % ext
assert 'cdnjs.cloudflare.com' not in h, 'ссылка на cdnjs уцелела'

open(html_out, 'w', encoding='utf-8').write(h)
print('katexify: %s (%d байт), require.js снят=%d, mermaid обезврежен=%d'
      % (html_out, len(h), n_req, n_mer), file=sys.stderr)
KATEXIFY_EOF

# --- 3. Печать с жёсткими предохранителями -----------------------------------
cat > printpdf.py <<'PRINT_EOF'
#!/usr/bin/env python3
"""HTML -> PDF. PDF не пишется вообще, если математика не отрисовалась.
usage: printpdf.py IN.html OUT.pdf EXPECTED_FORMULA_COUNT
Падает при: нет маркера / маркер != ok / есть .katex-error / число формул !=
ожидаемого / была хоть одна попытка сетевого запроса."""
import pathlib, sys
from playwright.sync_api import sync_playwright

html_in  = pathlib.Path(sys.argv[1]).resolve()
pdf_out  = pathlib.Path(sys.argv[2]).resolve()
expected = int(sys.argv[3])

offsite = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.on('request', lambda r: None if r.url.startswith('file:') else offsite.append(r.url))
    page.goto(html_in.as_uri(), wait_until='domcontentloaded', timeout=120_000)
    page.wait_for_function(
        "document.documentElement.getAttribute('data-math-ready') !== null", timeout=120_000)
    state  = page.evaluate("document.documentElement.getAttribute('data-math-ready')")
    count  = int(page.evaluate("document.documentElement.getAttribute('data-math-count')"))
    errors = int(page.evaluate("document.documentElement.getAttribute('data-math-errors')"))
    bad = []
    if state != 'ok':     bad.append('data-math-ready=%r' % state)
    if errors:            bad.append('%d узлов .katex-error' % errors)
    if count != expected: bad.append('отрисовано %d формул, ожидалось %d' % (count, expected))
    if offsite:           bad.append('были сетевые запросы: %r' % offsite[:5])
    if bad:
        browser.close()
        sys.exit('PDF НЕ ЗАПИСАН: ' + '; '.join(bad))
    print('katex nodes: %d, errors: 0, сетевых запросов: 0' % count, file=sys.stderr)
    page.pdf(path=str(pdf_out), prefer_css_page_size=True, print_background=True, format='Letter')
    browser.close()
print('wrote %s' % pdf_out, file=sys.stderr)
PRINT_EOF

# --- 4. Конвейер по обоим ноутбукам ------------------------------------------
build () {                       # build <ipynb> <имя без расширения> <куда положить>
  local src="$1"
  local stem="$2"
  local dest="$3"
  local d="$WORK/build/$stem"     # отдельная строка: в zsh с set -u $stem ещё не виден
  rm -rf "$d"; mkdir -p "$d"
  # промежуточные файлы называем как итог: nbconvert кладёт имя в <title>,
  # а Chrome копирует <title> в метаданные PDF (иначе там окажется "_prep").
  python3 "$WORK/prep.py" "$src" "$d/$stem.ipynb"
  python3 -m nbconvert --to html --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_input_tags='["remove_input"]' --no-prompt \
    --output-dir "$d" --output "$stem.html" "$d/$stem.ipynb"
  python3 "$WORK/katexify.py" "$d/$stem.html" "$d/$stem.katex.html" "$KATEX_DIST"
  python3 "$WORK/printpdf.py" "$d/$stem.katex.html" "$dest/$stem.pdf" "$NFORMULAS"
}

build "$REPO/notebooks/ru/eagle3-qwen3.ipynb"    eagle3-qwen3    "$REPO/notebooks/ru"
build "$REPO/notebooks/en/eagle3-qwen3-en.ipynb" eagle3-qwen3-en "$REPO/notebooks/en"

# --- 5. Самопроверка ----------------------------------------------------------
for pdf in "$REPO/notebooks/ru/eagle3-qwen3.pdf" "$REPO/notebooks/en/eagle3-qwen3-en.pdf"; do
  print "\n===== $pdf"
  pdftotext "$pdf" "$WORK/_check.txt"
  print "  страниц:   $(pdfinfo "$pdf" | awk '/^Pages/{print $2}')  (ожидание: ru 34, en 33)"
  print "  картинок:  $(pdfimages -list "$pdf" | tail -n +3 | wc -l | tr -d ' ')  (должно быть 15)"
  for s in '\frac' '\mathrm' '\qquad' '\bigl' '\approx' '\cdot' '\tau' '$$' '$'; do
    printf '  %-10s %s (должно быть 0)\n' "$s" "$(grep -F -o -- "$s" "$WORK/_check.txt" | wc -l | tr -d ' ')"
  done
  for s in 'measure_tree' 'plt.subplots' 'import os, sys, gc'; do
    printf '  %-20s %s (должно быть 0)\n' "$s" "$(grep -F -o -- "$s" "$WORK/_check.txt" | wc -l | tr -d ' ')"
  done
  printf '  tau        %s (должно быть >0)\n' "$(grep -F -o 'τ' "$WORK/_check.txt" | wc -l | tr -d ' ')"
  printf '  TFLOPS     %s (должно быть >0)\n' "$(grep -F -o 'TFLOPS' "$WORK/_check.txt" | wc -l | tr -d ' ')"
  printf '  op/param   %s (должно быть >0)\n' "$(grep -F -o 'op/param' "$WORK/_check.txt" | wc -l | tr -d ' ')"
done
# =============================================================================
# Прогнан целиком в свежем $WORK, exit 0, обе самопроверки чистые:
#   ru 34 стр / 15 картинок / 18 узлов .katex / 0 .katex-error / 0 сетевых запросов
#   en 33 стр / 15 картинок / 18 узлов .katex / 0 .katex-error / 0 сетевых запросов
# Текстовый слой побайтово совпадает с тем, что даёт исходный вариант конвейера.
# =============================================================================