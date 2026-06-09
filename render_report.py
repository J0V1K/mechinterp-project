"""Render report.md to report_render.html and report.pdf.

Pipeline:
  report.md  --[python-markdown + extensions]-->  report_render.html
                                                       |
                                                       v
                                                  [weasyprint] --> report.pdf

Run:  python render_report.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import markdown


HERE = Path(__file__).resolve().parent  # absolute, so file:// URLs work
SRC = HERE / "report.md"
HTML = HERE / "report_render.html"
PDF = HERE / "report.pdf"

CSS = """\
@page { size: Letter; margin: 0.7in; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  color: #111;
  line-height: 1.42;
  font-size: 11pt;
  max-width: 8in;
  margin: 0 auto;
}
h1, h2, h3 { line-height: 1.2; margin-top: 1.15em; margin-bottom: 0.45em; }
h1 { font-size: 24pt; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
h2 { font-size: 16pt; }
h3 { font-size: 13pt; }
p, ul, ol, table { margin-top: 0.45em; margin-bottom: 0.7em; }
ul, ol { padding-left: 1.2em; }
li { margin: 0.15em 0; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; }
th, td { border: 1px solid #d0d7de; padding: 6px 8px; vertical-align: top; }
th { background: #f6f8fa; }
img { max-width: 100%; height: auto; display: block; margin: 0.6em auto 0.4em; page-break-inside: avoid; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; background: #f6f8fa; padding: 0.1em 0.25em; }
pre { background: #f6f8fa; padding: 0.8em; overflow-x: auto; border-radius: 4px; }
pre code { background: transparent; padding: 0; }
blockquote { border-left: 4px solid #d0d7de; margin-left: 0; padding-left: 1em; color: #444; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.2em 0; }
"""


def main() -> int:
    md_text = SRC.read_text()
    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html = (
        "<!doctype html>\n<html>\n<head>\n"
        '<meta charset="utf-8" />\n'
        "<title>Shuffling Subliminal Sequences</title>\n"
        f"<style>\n{CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body_html}\n"
        "</body>\n</html>\n"
    )
    HTML.write_text(html)
    print(f"wrote {HTML}")

    # weasyprint on macOS often misses the gobject library; fall back to
    # headless Chrome which is universally available.
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # Note: `--print-to-pdf-no-header` is the working incantation here.
    # `--no-pdf-header-footer` parses but yields a text-only PDF that omits
    # all embedded images.
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={PDF}",
            f"file://{HTML}",
        ],
        check=True,
    )
    print(f"wrote {PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
