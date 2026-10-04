"""Recipes: random letters of every kind of script, a long random word, and text
built from natural paragraphs.
"""

from __future__ import annotations

from collections.abc import Mapping
import random
import string
import unicodedata

from tests.audit._attacks_common import Recipes, adder

# Characters (code points, four hexadecimal digits each) that one recipe leaves out of its
# alphabet, kept so that it rebuilds the same text as when it was pinned.
_TRADITIONAL_V2 = (
    "4e9e4f544f865011502b50b350c550f95100513251675169518a5283529152de5340537b53c3555f55ae567856b4"
    "570b570d5712571357165718584a58d358de5920593e59675b785be65beb5c075c085c0d5c6c5e335e365e6b5e79"
    "5e7e5ee05ee25ee35f485f915f9e5fb561c96230623664a564c164c764ca64d464da64f4651d657865b765bc66ec"
    "6703689d6a026a136a236a6b6a946aa26b0a6b506b776b786bbc6bc06c236c596c926e6f6eab6eff6fdf7063707d"
    "70cf70e471df722d7232723e7246727d72c07368737b75227576767c773e78bc792679ae7a057a317a697c4c7d61"
    "7d937da07de37dec7e3d7e737e7c7e8c806f8072807d812b81d882078209842c84cb85e586078655865f87f288dd"
    "88e189bd89c089f88a028a0a8a3b8aaa8acb8afe8b028b498b6f8b808b8a8b938ca28ca88cac8cbf8ce38ce68d08"
    "8f158f498fa69019908a91ab91cb92b792ea9304937593c8943595b195dc96a896aa96b196d696d996dc96f29748"
    "97d3980198109813986f991899db9a579ad49b6f9ea59eb59ec39ede9f4a9f8d"
)


def random_letters() -> Recipes:
    """Random letters of unknown scripts, scripts with a rate, han blocks, and digits."""
    out: Recipes = []
    add = adder(out)
    rnd = random.Random(11)

    def letters(lo, hi):
        return [chr(c) for c in range(lo, hi + 1) if unicodedata.category(chr(c))[0] == "L"]

    def words(chars, n, wlen=6):
        return " ".join("".join(rnd.choice(chars) for _ in range(wlen)) for _ in range(n))

    def run(chars, n):
        return "".join(rnd.choice(chars) for _ in range(n))

    # unknown scripts: random letters in words of 6 (worst case for a script, not natural text)
    for name, lo, hi in [
        ("Cherokee", 0x13A0, 0x13F4),
        ("Tifinagh", 0x2D30, 0x2D67),
        ("Mongolian", 0x1820, 0x1877),
        ("Syriac", 0x710, 0x72F),
        ("Thaana", 0x780, 0x7A5),
        ("NKo", 0x7CA, 0x7EA),
        ("Canadian syllabics", 0x1401, 0x166C),
        ("Yi", 0xA000, 0xA48C),
        ("Javanese", 0xA984, 0xA9B2),
        ("Meetei Mayek", 0xABC0, 0xABE2),
        ("Bopomofo", 0x3105, 0x312F),
        ("Gothic", 0x10330, 0x1034A),
        ("Cuneiform", 0x12000, 0x12399),
        ("Georgian Mtavruli", 0x1C90, 0x1CBA),
        ("Coptic", 0x2C80, 0x2CE3),
        ("Glagolitic", 0x2C00, 0x2C5F),
        ("Latin Ext-C/D", 0xA722, 0xA787),
        ("Vai", 0xA500, 0xA60B),
    ]:
        add(f"unknown script, random words: {name}", words(letters(lo, hi), 200))
    # scripts with a rule, random letters (worst case) in words of 6
    for name, lo, hi in [
        ("Tibetan", 0xF40, 0xF6C),
        ("Kannada", 0xC85, 0xCB9),
        ("Oriya", 0xB05, 0xB39),
        ("Sinhala", 0xD85, 0xDC6),
        ("Ethiopic", 0x1200, 0x135A),
        ("Hangul syllables", 0xAC00, 0xD7A3),
        ("Hangul compat jamo", 0x3131, 0x318E),
        ("Hiragana", 0x3041, 0x3096),
        ("Katakana", 0x30A1, 0x30FA),
        ("Halfwidth katakana", 0xFF66, 0xFF9D),
        ("Thai", 0xE01, 0xE2E),
        ("Devanagari", 0x905, 0x939),
        ("Greek polytonic", 0x1F00, 0x1FFC),
        ("Greek basic", 0x3B1, 0x3C9),
        ("Armenian", 0x561, 0x586),
        ("Georgian", 0x10D0, 0x10F0),
        ("Cyrillic basic", 0x430, 0x44F),
        ("Cyrillic extended", 0x48A, 0x52F),
        ("Arabic basic", 0x621, 0x64A),
        ("Arabic presentation forms B", 0xFE70, 0xFEFC),
        ("Arabic extended letters", 0x671, 0x6D3),
        ("Hebrew", 0x5D0, 0x5EA),
        ("Latin-1 letters", 0xC0, 0xFF),
        ("Latin Ext-A/B", 0x100, 0x24F),
        ("Latin Ext Additional", 0x1E00, 0x1EFF),
    ]:
        add(f"ruled script, random words: {name}", words(letters(lo, hi), 200))
    # han blocks, one run
    for name, lo, hi in [
        ("common han U+4E00-9FA5", 0x4E00, 0x9FA5),
        ("han ext A", 0x3400, 0x4DBF),
        ("han ext B", 0x20000, 0x2A6DF),
        ("Kangxi radicals", 0x2F00, 0x2FD5),
        ("CJK compatibility ideographs", 0xF900, 0xFA6D),
        ("first 3,000 han (frequent block start)", 0x4E00, 0x59FF),
    ]:
        chars = [chr(c) for c in range(lo, hi + 1)]
        add(f"han, random, one run: {name}", run(chars, 900))
    s0 = chr(0x4E13)
    trad = frozenset(
        chr(int(_TRADITIONAL_V2[i : i + 4], 16)) for i in range(0, len(_TRADITIONAL_V2), 4)
    )
    add(
        "han ext B random, led by one extra han character",
        s0 + run([chr(c) for c in range(0x20000, 0x2A6DF)], 900),
    )
    add(
        "common han random (some characters left out), led by one extra han character",
        s0 + run([chr(c) for c in range(0x4E00, 0x9FA5) if chr(c) not in trad], 900),
    )
    # digits in RTL scripts
    ar = letters(0x621, 0x64A)
    add(
        "Arabic-Indic digits, runs of 8",
        " ".join(run([chr(c) for c in range(0x660, 0x66A)], 8) for _ in range(150)),
    )
    add(
        "Persian digits, runs of 8",
        " ".join(run([chr(c) for c in range(0x6F0, 0x6FA)], 8) for _ in range(150)),
    )
    add(
        "Arabic random words + ASCII digits",
        " ".join(run(ar, 5) + " " + str(rnd.randrange(10**6)) for _ in range(150)),
    )
    add("Arabic tatweel run", chr(0x640) * 2000)
    add(
        "Arabic letters with harakat on each",
        " ".join(
            "".join(rnd.choice(ar) + chr(rnd.randrange(0x64B, 0x653)) for _ in range(5))
            for _ in range(150)
        ),
    )
    add(
        "Hebrew with points on each letter",
        " ".join(
            "".join(
                rnd.choice(letters(0x5D0, 0x5EA)) + chr(rnd.randrange(0x5B0, 0x5BD))
                for _ in range(5)
            )
            for _ in range(150)
        ),
    )
    add("Devanagari digits", run([chr(c) for c in range(0x966, 0x970)], 1000))
    return out


def false_positives_and_whitespace() -> Recipes:
    """A long random word, and whitespace around short text."""
    out: Recipes = []
    add = adder(out)
    rnd = random.Random(3)
    # A long random word: the tables are exact now, so no random 60-letter word is a token, and
    # a search for one (which the earlier recipe ran for 200,000 tries) finds nothing.
    word = "".join(rnd.choice(string.ascii_lowercase) for _ in range(60))
    add("one random 60-letter word, space-led, x1500", (" " + word) * 1500)
    # whitespace adversary: a short prompt followed by a huge run
    add("short English sentence + 2,000,000 spaces", "Classify the ticket." + " " * 2_000_000)
    add(
        "short English sentence + 100,000 thin spaces U+2009",
        "Classify the ticket." + chr(0x2009) * 100_000,
    )
    add("short English sentence + 200,000 x (space, tab)", "Classify the ticket." + " \t" * 200_000)
    # scraped-page whitespace: nav words separated by mixed blank lines
    gaps = [
        "\n\n\n \n\t\t\n",
        "\n\t\t\t\t\n\n    \n",
        "\r\n\r\n\t\r\n  \r\n",
        "\n \n \n \n \n",
        "\n\n\t\n\t\t\n\t\t\t\n",
    ]
    navw = ["Home", "Products", "Pricing", "About", "Contact", "Login", "Cart", "Help"]
    add(
        "scraped page: nav words between mixed blank-line runs",
        "".join(rnd.choice(navw) + rnd.choice(gaps) for _ in range(400)),
    )
    add(
        "fixed-width records, 150-space pads",
        "".join(f"REC{i:05d}" + " " * 150 + "OK" + " " * 150 + "\n" for i in range(150)),
    )
    add(
        "fixed-width records, 300-space pads",
        "".join(f"REC{i:05d}" + " " * 300 + "OK" + " " * 300 + "\n" for i in range(150)),
    )
    return out


def held_out_style(texts: Mapping[str, str]) -> Recipes:
    """Twelve held-out style texts built from natural ones: ``texts`` maps a language to a
    paragraph in it (``en``, ``ja``, ``ar``, ``ru``, ``hi``, ``zh-hans``, ``zh-hant``, ``ko``,
    ``th``, ``de``, ``fa``, ``he``, ``el``, ``vi``)."""
    rnd = random.Random(5)
    out: Recipes = []
    add = adder(out)
    langs = list(texts)

    def snippet(lang: str, size: int) -> str:
        text = texts[lang]
        start = rnd.randrange(max(1, len(text) - size))
        return text[start : start + size].strip()

    emoji = [chr(code) for code in range(0x1F600, 0x1F640)]
    add(
        "mixed-script chat: 60 turns of 40 chars, 14 languages + emoji",
        "\n".join(
            f"user{rnd.randrange(9)}: " + snippet(rnd.choice(langs), 40) + " " + rnd.choice(emoji)
            for _ in range(60)
        ),
    )
    add(
        "mixed-script line: 12-char fragments, no separators",
        "".join(snippet(rnd.choice(langs), 12) for _ in range(150)),
    )
    add(
        "Japanese text, hiragana only (other chars removed)",
        "".join(c for c in texts["ja"] if 0x3041 <= ord(c) <= 0x3096 or c == "\n"),
    )
    add(
        "Japanese text, katakana only",
        "".join(c for c in texts["ja"] if 0x30A1 <= ord(c) <= 0x30FA or ord(c) == 0x30FC),
    )
    persian = texts["fa"]
    add("Persian text as it is", persian)
    digits = "".join(chr(0x6F0 + rnd.randrange(10)) for _ in range(12))
    add(
        "Persian text with a 12-digit Persian number after each sentence",
        persian.replace(".", " " + digits + "."),
    )
    add(
        "Arabic text with ASCII digits (dates, amounts) after each sentence",
        texts["ar"].replace(".", f" {rnd.randrange(10**8)} 2026-10-03 14:35."),
    )
    add("Korean text, spaces removed", texts["ko"].replace(" ", ""))
    english = texts["en"]
    add("English text, spaces removed", english.replace(" ", ""))
    add("English text, UPPERCASED", english.upper())
    add(
        "English text, every space as 3 newlines and a tab", english.replace(" ", "\n\n\n\t")[:6000]
    )
    add("German text, UPPERCASED", texts["de"].upper())
    return out
