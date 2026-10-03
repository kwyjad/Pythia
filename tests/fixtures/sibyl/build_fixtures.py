# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regenerate the Sibyl reader fixtures: python tests/fixtures/sibyl/build_fixtures.py

The PDF is written by hand (no PDF library is a dependency): a few pages of
Helvetica text, one line per page, enough for pdfplumber to read back.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

PAGES = [
    "Cover page: Humanitarian Situation Report",
    "Contents and acknowledgements",
    "Weather outlook for the coming season",
    "Ethiopia: 1,250 people killed in clashes in Amhara during August 2026",
    "Logistics and pipeline status",
    "Annex: funding requirements",
]


def _pdf(pages: list[str]) -> bytes:
    objs: list[bytes] = []
    n = len(pages)
    # 1 catalog, 2 pages, 3 font, then (page, content) pairs.
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


HTML = """<html><head><title>Sitrep</title><script>var x=1;</script></head>
<body>
<nav>Home | About | Donate</nav>
<header>Site header</header>
<main>
<h1>Ethiopia: Situation Report</h1>
<p>Clashes continued in Amhara region.</p>
<table>
<tr><th>Region</th><th>Killed</th><th>Displaced</th></tr>
<tr><td>Amhara</td><td>1,250</td><td>40,000</td></tr>
</table>
<form>Subscribe <input name="e"></form>
</main>
<footer>Copyright footer</footer>
</body></html>
"""

if __name__ == "__main__":
    (HERE / "report.pdf").write_bytes(_pdf(PAGES))
    (HERE / "report.html").write_text(HTML, encoding="utf-8")
    print("written")
