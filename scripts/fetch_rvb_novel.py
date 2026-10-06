"""Загрузка текста романа с rvb.ru в TSV (verse_id, text) для независимой проверки.

Берутся страницы глав (<div class="chapter"> с <p id="pN">), абзац = <p>. Идентификатор
абзаца: <префикс>.<страница>.<номер на странице>, например BK.35_1-01.12.

Сырой HTML кладётся в кэш (по умолчанию /tmp/rvb_cache) и в репозиторий не
попадает; запросы идут по одному с паузой, чтобы не нагружать сайт.

Использование: fetch_rvb_novel.py <toc_url> <шаблон_ссылок_regex> <префикс> <out.tsv>
или:           fetch_rvb_novel.py --range <шаблон_url_с_{}> <от> <до> <префикс> <out.tsv> [доп_url ...]
пример: fetch_rvb_novel.py https://rvb.ru/dostoevski/tocvol9.htm "01text/vol9/35_1-\d+\.htm" BK out.tsv
"""
from __future__ import annotations

import html
import os
import re
import subprocess
import sys
import time

CACHE = os.environ.get("RVB_CACHE", "/tmp/rvb_cache")
UA = "Mozilla/5.0 (research; intertext)"


def get(url: str, allow_missing: bool = False) -> str | None:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, re.sub(r"[^\w.-]", "_", url))
    if os.path.exists(path):
        return open(path, "rb").read().decode("windows-1251", errors="replace")
    time.sleep(0.6)
    data = b""
    for attempt in range(3):
        # curl, а не urllib: встроенный в системный Python SSL с rvb.ru не договаривается
        data = subprocess.run(["curl", "-sL", "-m", "60", "-A", UA, url],
                              capture_output=True).stdout
        if len(data) > 3000 and b"<title>404" not in data[:2000]:
            break
        time.sleep(2 * (attempt + 1))
    else:
        if allow_missing:
            return None
        raise SystemExit(f"не скачалось: {url}")
    open(path, "wb").write(data)
    return data.decode("windows-1251", errors="replace")


def paragraphs(page: str) -> list[str]:
    # у страниц-продолжений нет блока chapter: тогда текст начинается с первого абзаца с id
    # (раньше такие страницы молча давали ноль абзацев -- в «Бесах» терялось три четверти текста)
    start = page.find('<div class="chapter"')
    if start < 0:
        start = page.find('<p id="p')
    body = page[max(start, 0):]
    out = []
    for m in re.finditer(r'<p id="p\d+"[^>]*>(.*?)</p>', body, re.S):
        t = re.sub(r"<br\s*/?>", " ", m.group(1))
        t = re.sub(r"<[^>]+>", "", t)
        t = html.unescape(t)
        t = re.sub(r"\[\d+\]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > 1:
            out.append(t)
    return out


def main(toc_url: str, link_pattern: str, prefix: str, out_tsv: str) -> None:
    toc = get(toc_url)
    base = toc_url.rsplit("/", 1)[0].rsplit("/", 1)[0] + "/"
    pages = []
    for h in re.findall(r'href="([^"]+)"', toc):
        if re.fullmatch(link_pattern, h.lstrip("/").replace("dostoevski/", "")) and h not in pages:
            pages.append(h)
    print(f"страниц глав: {len(pages)}", file=sys.stderr)
    n = 0
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("verse_id\ttext\n")
        for h in pages:
            url = "https://rvb.ru/dostoevski/" + h.lstrip("/").replace("dostoevski/", "")
            page_id = os.path.basename(h).replace(".htm", "")
            for k, t in enumerate(paragraphs(get(url)), 1):
                f.write(f"{prefix}.{page_id}.{k}\t{t.replace(chr(9), ' ')}\n")
                n += 1
    print(f"абзацев: {n} -> {out_tsv}")


def main_range(url_template: str, start: int, end: int, prefix: str, out_tsv: str, extra: list[str]) -> None:
    """Страницы по шаблону с номером (внутри романа они идут подряд, а в оглавлении
    есть только начала глав); отсутствующие номера пропускаются."""
    urls = [url_template.format(i) for i in range(start, end + 1)] + extra
    n = pages = 0
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("verse_id\ttext\n")
        for url in urls:
            page = get(url, allow_missing=True)
            if page is None:
                continue
            pages += 1
            page_id = os.path.basename(url).replace(".htm", "")
            for k, t in enumerate(paragraphs(page), 1):
                f.write(f"{prefix}.{page_id}.{k}\t{t.replace(chr(9), ' ')}\n")
                n += 1
    print(f"страниц {pages} из {len(urls)}, абзацев: {n} -> {out_tsv}")


if __name__ == "__main__":
    if sys.argv[1] == "--range":
        main_range(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7:])
    else:
        main(*sys.argv[1:5])
