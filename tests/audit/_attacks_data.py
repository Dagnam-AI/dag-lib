"""Recipes: encodings, structured data, logs, long identifiers and long words."""

from __future__ import annotations

import base64
import hashlib
import json
import random
import string

from tests.audit._attacks_common import Recipes, adder


def encodings_and_data() -> Recipes:
    """Base64 and the like, minified JSON and code, logs, URLs, long identifiers and words."""
    out: Recipes = []
    add = adder(out)
    rnd = random.Random(20261003)

    blob = bytes(rnd.randrange(256) for _ in range(6000))
    add("base64 one line", base64.b64encode(blob).decode())
    add("base64 url-safe no padding", base64.urlsafe_b64encode(blob).decode().rstrip("="))
    add(
        "base64 wrapped 76",
        "\n".join(base64.b64encode(blob).decode()[i : i + 76] for i in range(0, 8000, 76)),
    )
    add("data URL png", "data:image/png;base64," + base64.b64encode(blob).decode())
    add("base32", base64.b32encode(blob).decode())
    add("base32 lower", base64.b32encode(blob).decode().lower())
    add("hex lower", blob.hex())
    add("hex upper", blob.hex().upper())
    add("hex 0x.. bytes", ", ".join(f"0x{b:02x}" for b in blob[:1500]))
    add("base85", base64.b85encode(blob).decode())
    add("ascii85", base64.a85encode(blob).decode())
    add("percent-encoded", "".join(f"%{b:02X}" for b in blob[:2000]))
    add(
        "json \\u escapes",
        json.dumps("".join(chr(rnd.randrange(0x4E00, 0x9FFF)) for _ in range(800))),
    )

    # minified JSON / JS
    obj = [
        {
            "id": i,
            "sku": f"SKU-{rnd.randrange(10**6):06d}",
            "qty": rnd.randrange(50),
            "price": round(rnd.random() * 500, 2),
            "tags": ["a", "bb", "ccc"],
            "ok": True,
            "note": None,
            "addr": {"zip": f"{rnd.randrange(10**5):05d}", "cc": "DE"},
        }
        for i in range(120)
    ]
    add("minified JSON", json.dumps(obj, separators=(",", ":")))
    add("pretty JSON indent 8", json.dumps(obj[:40], indent=8))
    add("pretty JSON tabs", json.dumps(obj[:40], indent="\t"))
    js = "".join(
        f"function {a}({b},{c}){{var {d}={b}?{c}[{b}]:void 0;return {d}&&{d}.{a}||(e[{b}]={{}}),!0}}"
        for a, b, c, d in [
            (
                rnd.choice("abcdefgh") + rnd.choice("xyz"),
                rnd.choice("ijk"),
                rnd.choice("mnp"),
                rnd.choice("rst"),
            )
            for _ in range(120)
        ]
    )
    add("minified JS", js)
    add(
        "regexes",
        " ".join(
            r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@(?:[\w-]+\.)+[\w-]{2,}$"
            for _ in range(25)
        ),
    )
    add(
        "random ascii punctuation",
        "".join(rnd.choice("!@#$%^&*()-+=[]{}|;:<>,.?/~`\\\"'") for _ in range(3000)),
    )
    add(
        "random ascii punctuation, spaced",
        " ".join(
            "".join(rnd.choice("!@#$%^&*()-+=[]{}|;:<>,.?/~`\\") for _ in range(3))
            for _ in range(900)
        ),
    )
    add("brainf*ck", "".join(rnd.choice("+-<>[].,") for _ in range(3000)))
    add(
        "ascii table borders",
        "\n".join(
            "+" + "+".join("-" * rnd.randrange(3, 30) for _ in range(6)) + "+" for _ in range(60)
        ),
    )
    add(
        "md table",
        "\n".join(
            "| " + " | ".join(f"{rnd.random() * 1000:.2f}" for _ in range(6)) + " |"
            for _ in range(60)
        ),
    )
    add("tsv", "\n".join("\t".join(str(rnd.randrange(10**6)) for _ in range(8)) for _ in range(80)))

    # logs
    add(
        "logs ISO timestamps",
        "\n".join(
            f"2026-10-03T{rnd.randrange(24):02d}:{rnd.randrange(60):02d}:{rnd.randrange(60):02d}.{rnd.randrange(1000):03d}Z [worker-{rnd.randrange(16)}] INFO  com.acme.billing.InvoiceService - processed invoice id={rnd.randrange(10**9)} in {rnd.randrange(900)}ms"
            for _ in range(40)
        ),
    )
    add(
        "syslog",
        "\n".join(
            f"Oct  3 {rnd.randrange(24):02d}:{rnd.randrange(60):02d}:{rnd.randrange(60):02d} ip-10-0-{rnd.randrange(255)}-{rnd.randrange(255)} sshd[{rnd.randrange(65535)}]: Failed password for invalid user admin from {rnd.randrange(255)}.{rnd.randrange(255)}.{rnd.randrange(255)}.{rnd.randrange(255)} port {rnd.randrange(65535)} ssh2"
            for _ in range(40)
        ),
    )
    add(
        "ANSI colour log",
        "\n".join(
            f"\x1b[3{rnd.randrange(8)};1m[{i:04d}]\x1b[0m \x1b[2mstep\x1b[0m ok" for i in range(200)
        ),
    )

    # urls
    add(
        "URLs with query strings",
        "\n".join(
            f"https://shop.example.com/p/{rnd.randrange(10**8)}/checkout?utm_source=newsletter&utm_campaign=q{rnd.randrange(4)}_sale&session={hashlib.md5(str(i).encode(), usedforsecurity=False).hexdigest()}&redirect=https%3A%2F%2Fexample.com%2Fcart%3Fid%3D{rnd.randrange(10**6)}&token={base64.urlsafe_b64encode(bytes(rnd.randrange(256) for _ in range(24))).decode()}"
            for i in range(30)
        ),
    )

    # long identifiers
    ids = [
        "AbstractSingletonProxyFactoryBeanGeneratorStrategyImplementation",
        "kCFStreamPropertySOCKSProxyPasswordAuthenticationHandler",
        "get_customer_billing_address_validation_error_message_template",
        "MAX_CONCURRENT_STREAMING_CONNECTIONS_PER_TENANT_OVERRIDE",
        "internalFrameInternalFrameTitlePaneInternalFrameTitlePaneMaximizeButtonWindowNotFocusedState",
        "x-amz-server-side-encryption-customer-algorithm",
    ]
    add(
        "code, long identifiers",
        "\n".join(
            f"    private final {rnd.choice(ids)} {rnd.choice(ids)[0].lower()}{rnd.choice(ids)[1:]} = new {rnd.choice(ids)}({rnd.choice(ids)}.{rnd.choice(ids)});"
            for _ in range(40)
        ),
    )
    add(
        "SCREAMING_SNAKE list",
        " ".join(
            "_".join(
                rnd.choice(
                    [
                        "MAX",
                        "TENANT",
                        "OVERRIDE",
                        "STREAMING",
                        "CONNECTIONS",
                        "RETRY",
                        "BACKOFF",
                        "JITTER",
                        "QUOTA",
                    ]
                )
                for _ in range(5)
            )
            for _ in range(150)
        ),
    )
    add(
        "lowercase concatenated words",
        " ".join(
            "".join(
                rnd.choice(
                    [
                        "quick",
                        "brown",
                        "customer",
                        "invoice",
                        "refund",
                        "status",
                        "shipping",
                        "address",
                        "payment",
                    ]
                )
                for _ in range(6)
            )
            for _ in range(120)
        ),
    )
    add(
        "hashtags",
        " ".join(
            "#"
            + "".join(
                rnd.choice(
                    [
                        "Summer",
                        "Sale",
                        "Black",
                        "Friday",
                        "Deal",
                        "Free",
                        "Shipping",
                        "New",
                        "Arrivals",
                    ]
                )
                for _ in range(4)
            )
            for _ in range(200)
        ),
    )

    # very long single words
    low = "".join(rnd.choice(string.ascii_lowercase) for _ in range(4000))
    add("one long word, random lowercase", low)
    add("one long word, random UPPERCASE", low.upper())
    add(
        "one long word, random mixed case",
        "".join(rnd.choice(string.ascii_letters) for _ in range(4000)),
    )
    add(
        "random lowercase words 8-12 letters",
        " ".join(
            "".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randrange(8, 13)))
            for _ in range(300)
        ),
    )
    add(
        "random UPPERCASE words 8-12 letters",
        " ".join(
            "".join(rnd.choice(string.ascii_uppercase) for _ in range(rnd.randrange(8, 13)))
            for _ in range(300)
        ),
    )
    add(
        "random UPPERCASE words 20-40 letters",
        " ".join(
            "".join(rnd.choice(string.ascii_uppercase) for _ in range(rnd.randrange(20, 41)))
            for _ in range(120)
        ),
    )
    add("DNA sequence one word", "".join(rnd.choice("ACGT") for _ in range(4000)))
    add(
        "DNA FASTA lines of 60",
        "\n".join("".join(rnd.choice("ACGT") for _ in range(60)) for _ in range(60)),
    )
    add(
        "protein sequence one word",
        "".join(rnd.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(3000)),
    )
    add(
        "protein FASTA lines of 60",
        "\n".join(
            "".join(rnd.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(60)) for _ in range(50)
        ),
    )
    add(
        "license keys XXXXX-XXXXX",
        "\n".join(
            "-".join(
                "".join(rnd.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(5)
            )
            for _ in range(80)
        ),
    )
    add(
        "IUPAC names",
        " ".join(
            [
                "methylenedioxymethamphetamine",
                "hexahydroxycyclohexanehexacarboxylate",
                "tetrahydrocannabinolic",
                "dichlorodiphenyltrichloroethane",
                "pneumonoultramicroscopicsilicovolcanoconiosis",
                "polytetrafluoroethylene",
                "deoxyribonucleotidyltransferase",
            ]
            * 30
        ),
    )
    add(
        "consonant-only lowercase words",
        " ".join(
            "".join(rnd.choice("bcdfghjklmnpqrstvwxz") for _ in range(10)) for _ in range(300)
        ),
    )

    # repeated characters
    for ch, n in [
        ("a", 6000),
        ("A", 6000),
        ("z", 4000),
        ("=", 8000),
        ("-", 8000),
        (".", 8000),
        ("!", 6000),
        ("?", 4000),
        ("~", 3000),
        ("^", 3000),
        ("ab", 3000),
        ("xyz", 2000),
        ("ha", 3000),
        ("0", 2000),
        ("\u54c8", 3000),
        ("\u3002", 3000),
        ("\u2026", 3000),
        ("\u30fc", 3000),
        ("\u314b", 3000),
        ("\U0001f602", 1000),
        ("\u0301", 2000),
    ]:
        add(f"repeated {ch!r} x{n}", ch * n)
    return out
