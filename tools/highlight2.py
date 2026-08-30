"""Precise highlighting: red on placeholder text only, yellow on uncertain claims."""
import re, sys, zipfile, shutil, os, html

RED_INLINE = r"\[TO SUPPLY:[^\]]*\]"          # split out of its run, red only on this
RED_WHOLE = [                                  # template boilerplate, red whole run
    "The abstract is a formal description", "No part of this project has been submitted",
    "The acknowledgements thank the people", "Principal Components Analysis",
    "Three Letter Acronym", "Department of Computing and Mathematics",
    "First Chapter - probably", "Second Chapter - probably", "The second appendix",
    "Quality measurements used by CQA", "Transform coding",
]
YELLOW = [
    "This project takes PerMFL", "This matters more than a sanity check",
    "The comparison against other methods is absent",
    "Several exploratory results rest on single runs",
    "The clustering signal is parameter-space only",
    "The structural ceiling calculation assumes",
    "It transfers, but not as published",
    "One objective is only partly met",
    "No comparison is made against the two most closely related methods",
    "The loader interface materialises every client's features",
    "Sixteen defects were catalogued",
]
LEGEND = ("Draft marking: text highlighted in RED is a placeholder that must be replaced "
          "before submission. Text highlighted in YELLOW is a claim that is either "
          "unverified, rests on a single run, or depends on a citation whose details need "
          "checking. Neither colour should survive into the final document.")

def esc(t): return html.escape(t, quote=False)

def run_with(text, colour=None, extra=""):
    rpr = f'<w:rPr>{extra}{f"<w:highlight w:val=\"{colour}\"/>" if colour else ""}</w:rPr>' if (colour or extra) else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

def apply(path):
    work = "/tmp/hl2"; shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    with zipfile.ZipFile(path) as z: z.extractall(work)
    doc = os.path.join(work, "word/document.xml"); x = open(doc, encoding="utf8").read()
    n_red = n_yel = 0

    # 1. split [TO SUPPLY: ...] out of its run and colour only that fragment
    def split_placeholder(m):
        nonlocal n_red
        r = m.group(0)
        txt = "".join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', r))
        plain = html.unescape(txt)
        if not re.search(RED_INLINE, plain): return r
        pre_rpr = re.search(r'<w:rPr>.*?</w:rPr>', r, re.S)
        keep = pre_rpr.group(0)[7:-8] if pre_rpr else ""
        out, last = [], 0
        for mm in re.finditer(RED_INLINE, plain):
            if mm.start() > last: out.append(run_with(plain[last:mm.start()], None, keep))
            out.append(run_with(mm.group(0), "red", keep)); n_red += 1
            last = mm.end()
        if last < len(plain): out.append(run_with(plain[last:], None, keep))
        return "".join(out)
    x = re.sub(r'<w:r\b[^>]*>(?:(?!</w:r>).)*?</w:r>', split_placeholder, x, flags=re.S)

    # 2. whole-run highlights
    def mark(patterns, colour):
        nonlocal x, n_red, n_yel
        hits = []
        for m in re.finditer(r'<w:r\b[^>]*>(?:(?!</w:r>).)*?</w:r>', x, re.S):
            t = html.unescape("".join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', m.group(0))))
            if any(p in t for p in patterns) and "<w:highlight" not in m.group(0):
                hits.append(m)
        for m in reversed(hits):
            r = m.group(0)
            r2 = (r.replace("<w:rPr>", f'<w:rPr><w:highlight w:val="{colour}"/>', 1)
                  if "<w:rPr>" in r else
                  r.replace("<w:r>", f'<w:r><w:rPr><w:highlight w:val="{colour}"/></w:rPr>', 1))
            x = x[:m.start()] + r2 + x[m.end():]
        if colour == "red": n_red += len(hits)
        else: n_yel += len(hits)
    mark(RED_WHOLE, "red"); mark(YELLOW, "yellow")

    # 3. legend after the contents heading
    anchor = '<w:pStyle w:val="TOCHeading"/>'
    if anchor in x:
        idx = x.find("</w:p>", x.find(anchor)) + 6
        legend = (f'<w:p><w:pPr><w:pBdr><w:top w:val="single" w:sz="6" w:space="4" w:color="C00000"/>'
                  f'<w:bottom w:val="single" w:sz="6" w:space="4" w:color="C00000"/></w:pBdr>'
                  f'<w:spacing w:before="120" w:after="120"/></w:pPr>'
                  f'<w:r><w:rPr><w:i/><w:sz w:val="18"/></w:rPr>'
                  f'<w:t xml:space="preserve">{esc(LEGEND)}</w:t></w:r></w:p>')
        x = x[:idx] + legend + x[idx:]

    open(doc, "w", encoding="utf8").write(x)
    os.remove(path)
    zf = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
    for root, _, files in os.walk(work):
        for f in files:
            fp = os.path.join(root, f); zf.write(fp, os.path.relpath(fp, work))
    zf.close()
    print(f"  {os.path.basename(path)}: {n_red} red, {n_yel} yellow, legend added")

for a in sys.argv[1:]: apply(a)
