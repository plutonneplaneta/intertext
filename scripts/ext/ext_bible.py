# -*- coding: utf-8 -*-
"""Сокращения книг Библии (Баньян и др.) -> коды корпуса KJV; разбор скобки со ссылками.
Таблица составлена до просмотра результатов (prereg_material_C.md, правило 5)."""
import re
BOOKS = {
 'GEN': 'Genesis Gen', 'EXO': 'Exodus Exod Ex', 'LEV': 'Leviticus Lev', 'NUM': 'Numbers Num',
 'DEU': 'Deuteronomy Deut', 'JOS': 'Joshua Josh', 'JDG': 'Judges Judg', 'RUT': 'Ruth',
 '1SA': '1Samuel 1Sam', '2SA': '2Samuel 2Sam', '1KI': '1Kings 1Kgs', '2KI': '2Kings 2Kgs',
 '1CH': '1Chronicles 1Chron', '2CH': '2Chronicles 2Chron', 'EZR': 'Ezra', 'NEH': 'Nehemiah Neh',
 'EST': 'Esther Esth', 'JOB': 'Job', 'PSA': 'Psalm Psalms Ps', 'PRO': 'Proverbs Prov',
 'ECC': 'Ecclesiastes Eccl Eccles', 'SON': 'Song Cant Canticles', 'ISA': 'Isaiah Isa', 'JER': 'Jeremiah Jer',
 'LAM': 'Lamentations Lam', 'EZE': 'Ezekiel Ezek', 'DAN': 'Daniel Dan', 'HOS': 'Hosea Hos',
 'JOE': 'Joel', 'AMO': 'Amos', 'OBA': 'Obadiah Obad', 'JON': 'Jonah', 'MIC': 'Micah Mic', 'NAH': 'Nahum Nah',
 'HAB': 'Habakkuk Hab', 'ZEP': 'Zephaniah Zeph', 'HAG': 'Haggai Hag', 'ZEC': 'Zechariah Zech', 'MAL': 'Malachi Mal',
 'MAT': 'Matthew Matt', 'MAR': 'Mark', 'LUK': 'Luke', 'JOH': 'John', 'ACT': 'Acts', 'ROM': 'Romans Rom',
 '1CO': '1Corinthians 1Cor', '2CO': '2Corinthians 2Cor', 'GAL': 'Galatians Gal', 'EPH': 'Ephesians Eph',
 'PHI': 'Philippians Phil', 'COL': 'Colossians Col', '1TH': '1Thessalonians 1Thess',
 '2TH': '2Thessalonians 2Thess', '1TI': '1Timothy 1Tim', '2TI': '2Timothy 2Tim', 'TIT': 'Titus',
 'PHM': 'Philemon Philem', 'HEB': 'Hebrews Heb', 'JAM': 'James Jas', '1PE': '1Peter 1Pet', '2PE': '2Peter 2Pet',
 '1JO': '1John', '2JO': '2John', '3JO': '3John', 'JUD': 'Jude', 'REV': 'Revelation Revelations Rev',
}
ALIAS = {n.lower(): c for c, ns in BOOKS.items() for n in ns.split()}


def norm_book(s):
    return ALIAS.get(re.sub(r'[\s.]', '', s).lower())


BOOK_RE = r'(?:[123]\s?)?[A-Z][a-z]+'
ITEM = re.compile(r'^\s*(?:(\d+)\s*[:.]\s*)?(\d+)(?:\s*[-–]\s*(\d+))?\s*$')


def parse_bracket(inner):
    """-> список (book_code|None, book_text, chapter, v1, v2, status) или None, если скобка не ссылка.
    Скобка -- ссылка, если каждая часть через ';' начинается с названия книги или (после первой) с цифр."""
    inner = re.sub(r'\s+', ' ', inner).strip().rstrip('.')
    parts = [p.strip() for p in inner.split(';') if p.strip()]
    if not parts: return None
    out = []; book = None; btxt = None; chap = None
    for k, p in enumerate(parts):
        m = re.match(r'^((?:[123]\s?)?[A-Z][a-z]+\.?)\s+(.*)$', p)
        if m:
            btxt = m.group(1); book = norm_book(btxt); rest = m.group(2); chap = None
        elif k > 0 and book is not None and re.match(r'^\d', p):
            rest = p
        else:
            return None
        items = [x.strip() for x in rest.split(',') if x.strip()]
        if not items or not all(ITEM.match(x) for x in items): return None
        for x in items:
            mm = ITEM.match(x)
            if mm.group(1): chap = int(mm.group(1))
            if chap is None:
                out.append((book, btxt, None, int(mm.group(2)), None, 'NO_VERSE')); continue
            v1 = int(mm.group(2)); v2 = int(mm.group(3)) if mm.group(3) else v1
            out.append((book, btxt, chap, v1, max(v1, v2), 'ok' if book else 'NO_BOOK'))
    return out
