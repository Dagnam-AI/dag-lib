"""Recipes: whitespace, controls, format characters, emoji, marks, mathematical and fancy text."""

from __future__ import annotations

import random
import unicodedata

from tests.audit._attacks_common import Recipes, adder


def characters_and_whitespace() -> Recipes:
    """Whitespace, controls, format characters, emoji, marks, mathematical and fancy text."""
    out: Recipes = []
    add = adder(out)
    rnd = random.Random(7)

    def rc(lo, hi, n, sep=""):
        return sep.join(chr(rnd.randrange(lo, hi + 1)) for _ in range(n))

    def assigned(lo, hi, cats=None):
        return [
            chr(c)
            for c in range(lo, hi + 1)
            if unicodedata.category(chr(c)) != "Cn"
            and (cats is None or unicodedata.category(chr(c))[0] in cats)
        ]

    def pick(chars, n, word=0):
        if not word:
            return "".join(rnd.choice(chars) for _ in range(n))
        return " ".join("".join(rnd.choice(chars) for _ in range(word)) for _ in range(n // word))

    # whitespace
    add("spaces x200000", " " * 200000)
    add("tabs x20000", "\t" * 20000)
    add("newlines x20000", "\n" * 20000)
    add("CRLF x10000", "\r\n" * 10000)
    add("alternating space/tab x10000", " \t" * 5000)
    add("NBSP x3000", "\u00a0" * 3000)
    add("ideographic space x3000", "\u3000" * 3000)
    add("em/thin/hair spaces mix x3000", pick("\u2002\u2003\u2009\u200a\u202f\u205f", 3000))
    add("line/para separators x2000", "\u2028\u2029" * 1000)
    add("vertical tab/form feed x3000", "\x0b\x0c" * 1500)
    add("words padded with 40 spaces", (" " * 40).join(["total", "amount", "due"] * 300))
    add(
        "lines with trailing 80 spaces + CRLF",
        "".join(f"row {i}" + " " * 80 + "\r\n" for i in range(300)),
    )
    add("lines indented 7 tabs", "".join("\t" * 7 + "x\n" for _ in range(600)))
    add(
        "jp text with U+3000 indents",
        "".join("\u3000\u3000\u3000\u3000" + "\u6ce8\u610f\u4e8b\u9805" + "\n" for _ in range(300)),
    )

    # zero width / bidi / controls
    add("ZWSP x3000", "\u200b" * 3000)
    add("ZWJ x3000", "\u200d" * 3000)
    add("ZWNJ x3000", "\u200c" * 3000)
    add("BOM/WJ x3000", "\ufeff\u2060" * 1500)
    add(
        "bidi controls x3000",
        pick("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f", 3000),
    )
    add(
        "word ZWSP word (Khmer/Thai style breaks) latin",
        "\u200b".join(["hello", "world", "refund", "please"] * 250),
    )
    add(
        "soft hyphen in words",
        " ".join(
            "in\u00adter\u00adna\u00adtion\u00adal\u00adi\u00adza\u00adtion" for _ in range(150)
        ),
    )
    add("NUL x3000", "\x00" * 3000)
    add(
        "C0 controls mix x3000",
        pick([chr(c) for c in list(range(1, 9)) + list(range(14, 28))], 3000),
    )
    add("DEL x3000", "\x7f" * 3000)
    add("C1 controls x3000", rc(0x80, 0x9F, 3000))
    add("NUL-separated fields", "\x00".join(["alpha", "42", "beta"] * 400))
    add("U+FFFD x2000", "\ufffd" * 2000)
    add("noncharacters U+FFFE/FFFF x2000", "\ufffe\uffff" * 1000)
    add("lone surrogates x2000 [tokenizer: TypeError]", "\ud800" * 2000)
    add("private use BMP x2000", rc(0xE000, 0xF8FF, 2000))
    add("private use plane 15 x1000", rc(0xF0000, 0xFFFFD, 1000))
    add("unassigned U+0378.. x2000", "\u0378\u0379" * 1000)
    add("unassigned plane 3 x1000", rc(0x31400, 0x31FFF, 1000))
    add("tag characters x1000", rc(0xE0020, 0xE007E, 1000))
    add("variation selectors FE0F x2000", "\ufe0f" * 2000)

    # emoji
    emo = [chr(c) for c in range(0x1F600, 0x1F650)]
    add("emoji run random faces x600", pick(emo, 600))
    add("emoji spaced x600", " ".join(rnd.choice(emo) for _ in range(600)))
    add("ZWJ family x150", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466" * 150)
    add("ZWJ profession+skin tone x150", "\U0001f469\U0001f3fd\u200d\U0001f4bb" * 150)
    add(
        "flags x300",
        "".join(
            chr(0x1F1E6 + rnd.randrange(26)) + chr(0x1F1E6 + rnd.randrange(26)) for _ in range(300)
        ),
    )
    add(
        "subdivision flag (tags) x60",
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f" * 60,
    )
    add("keycaps x300", "".join(d + "\ufe0f\u20e3" for d in "0123456789" * 30))
    add("heart+VS16 x500", "\u2764\ufe0f" * 500)
    add(
        "BMP symbols dingbats/arrows x1500",
        pick(assigned(0x2190, 0x21FF) + assigned(0x2700, 0x27BF), 1500),
    )
    add("box drawing x2000", pick(assigned(0x2500, 0x257F), 2000))
    add(
        "box table",
        "\n".join(
            "\u2502 "
            + " \u2502 ".join(str(rnd.randrange(1000)) for _ in range(5))
            + " \u2502\n\u251c"
            + "\u253c".join("\u2500" * 6 for _ in range(5))
            + "\u2524"
            for _ in range(40)
        ),
    )
    add("braille x1500", rc(0x2800, 0x28FF, 1500))
    add(
        "emoji chat mixed",
        " ".join(
            rnd.choice(["ok", "lol", "thanks!!", "omg", "see you"])
            + " "
            + rnd.choice(emo)
            + rnd.choice(emo)
            for _ in range(250)
        ),
    )
    add("rare plane-1 pictographs x600", rc(0x1FA70, 0x1FAFF, 600))

    # combining
    add(
        "zalgo",
        "".join(
            ch + "".join(chr(rnd.randrange(0x300, 0x36F)) for _ in range(6))
            for ch in "the customer is always right " * 12
        ),
    )
    add(
        "NFD French",
        unicodedata.normalize(
            "NFD",
            "Les \u00e9l\u00e8ves ont \u00e9t\u00e9 tr\u00e8s \u00e9tonn\u00e9s \u00e0 l'id\u00e9e d'\u00eatre re\u00e7us pr\u00e8s de l'h\u00f4tel o\u00f9 na\u00eet le ma\u00efs. "
            * 30,
        ),
    )
    add(
        "NFD Vietnamese",
        unicodedata.normalize(
            "NFD",
            "T\u1ea5t c\u1ea3 m\u1ecdi ng\u01b0\u1eddi sinh ra \u0111\u1ec1u \u0111\u01b0\u1ee3c t\u1ef1 do v\u00e0 b\u00ecnh \u0111\u1eb3ng v\u1ec1 nh\u00e2n ph\u1ea9m v\u00e0 quy\u1ec1n l\u1ee3i. "
            * 25,
        ),
    )
    add(
        "NFD Korean (jamo)",
        unicodedata.normalize(
            "NFD",
            "\ubaa8\ub4e0 \uc778\uac04\uc740 \ud0dc\uc5b4\ub0a0 \ub54c\ubd80\ud130 \uc790\uc720\ub85c\uc6b0\uba70 \uadf8 \uc874\uc5c4\uacfc \uad8c\ub9ac\uc5d0 \uc788\uc5b4 \ub3d9\ub4f1\ud558\ub2e4. "
            * 15,
        ),
    )
    add(
        "NFC Korean (control)",
        "\ubaa8\ub4e0 \uc778\uac04\uc740 \ud0dc\uc5b4\ub0a0 \ub54c\ubd80\ud130 \uc790\uc720\ub85c\uc6b0\uba70 \uadf8 \uc874\uc5c4\uacfc \uad8c\ub9ac\uc5d0 \uc788\uc5b4 \ub3d9\ub4f1\ud558\ub2e4. "
        * 30,
    )
    add(
        "NFD Japanese",
        unicodedata.normalize(
            "NFD",
            "\u3059\u3079\u3066\u306e\u4eba\u9593\u306f\u3001\u751f\u307e\u308c\u306a\u304c\u3089\u306b\u3057\u3066\u81ea\u7531\u3067\u3042\u308a\u3001\u304b\u3064\u3001\u5c0a\u53b3\u3068\u6a29\u5229\u3068\u306b\u3064\u3044\u3066\u5e73\u7b49\u3067\u3042\u308b\u3002\u30ac\u30ae\u30b0\u30b2\u30b4"
            * 20,
        ),
    )
    add(
        "NFD German",
        unicodedata.normalize(
            "NFD",
            "\u00dcber die gr\u00f6\u00dften Sch\u00e4den m\u00fcssen \u00f6ffentliche \u00c4mter fr\u00fch h\u00f6ren. "
            * 40,
        ),
    )

    # math / fancy
    add(
        "LaTeX",
        r"\begin{align} \frac{\partial \mathcal{L}}{\partial \theta_{ij}} &= \sum_{k=1}^{N} \left( \hat{y}_k - y_k \right) \cdot \nabla_{\theta} f(x_k; \theta) + \lambda \lVert \theta \rVert_2^2 \\ \int_{-\infty}^{\infty} e^{-x^2} \, dx &= \sqrt{\pi} \end{align} "
        * 12,
    )
    add("unicode math operators x1500", pick(assigned(0x2200, 0x22FF), 1500))
    add(
        "unicode math sentence",
        "\u2200x\u2208\u211d \u2203y\u2208\u2115: x\u00b2+y\u00b2\u2264z\u00b3 \u2227 \u2211\u1d62\u208c\u2081\u207f a\u1d62\u00b7b\u1d62 \u2260 \u2205 \u21d2 \u222b\u2080^\u221e f(t)dt \u2248 \u03c0/\u221a2 \u00b1 \u03b5 "
        * 30,
    )
    add(
        "math bold letters (fancy text)",
        " ".join(
            "".join(chr(0x1D400 + (ord(c) - 65 if c.isupper() else ord(c) - 97 + 26)) for c in w)
            for w in ("Limited time offer buy now and save big today " * 25).split()
        ),
    )
    add("math script/fraktur random", rc(0x1D4D0, 0x1D537, 800))
    add(
        "fullwidth latin sentence",
        "".join(
            chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else "\u3000"
            for c in "Please confirm your order number and shipping address today. " * 15
        ),
    )
    add("fullwidth digits", "".join(chr(0xFF10 + rnd.randrange(10)) for _ in range(1500)))
    add("circled/enclosed alnum x1000", pick(assigned(0x2460, 0x24FF), 1000))
    add(
        "superscripts/subscripts x1000",
        pick(
            "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u207f\u1d62",
            1000,
        ),
    )
    add("roman numerals/fractions x1000", pick(assigned(0x2150, 0x218B), 1000))
    add(
        "small caps / IPA sentence",
        "\u00f0\u0259 \u02c8k\u028cst\u0259m\u0259 s\u025c\u02d0v\u026as \u02ccr\u025bpr\u026a\u02c8z\u025bnt\u0259t\u026av w\u026al k\u0259n\u02c8t\u00e6kt ju\u02d0 \u0283\u0254\u02d0tli \u0259\u02c8ba\u028at j\u0254\u02d0 \u02c8\u0254\u02d0d\u0259 "
        * 20,
    )
    add("IPA random", pick(assigned(0x250, 0x2AF), 1200, word=6))
    add(
        "upside-down text",
        "\u0287\u0265\u0183\u0131\u0279 s\u028e\u0250\u028dl\u0250 s\u0131 \u0279\u01dd\u026fo\u0287sn\u0254 \u01dd\u0265\u0287 "
        * 40,
    )
    add(
        "leet/small caps",
        "\u1d1b\u029c\u1d07 \u1d04\u1d1cs\u1d1b\u1d0f\u1d0d\u1d07\u0280 \u026as \u1d00\u029f\u1d21\u1d00\u028fs \u0280\u026a\u0262\u029c\u1d1b "
        * 40,
    )
    return out
