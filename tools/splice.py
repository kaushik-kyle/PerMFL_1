"""Rebuild a report from the MMU template, preserving cover sheet, styles,
front matter, footers and section properties. Body content is replaced."""
import re, shutil, zipfile, os, sys, html, struct

TMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "template", "report.docx")
md_file, out_file, title, subtitle = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
work = "/tmp/build_docx"
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
with zipfile.ZipFile(TMPL) as z: z.extractall(work)

doc = os.path.join(work, "word/document.xml")
x = open(doc, encoding="utf8").read()
paras = list(re.finditer(r'<w:p\b[^>]*>.*?</w:p>|<w:p\b[^>]*/>', x, re.S))

# keep paragraphs 0..38 (cover, TOC, lists, abstract..abbreviations, section break)
KEEP_TO = 38
head_end = paras[KEEP_TO].end()
tail_start = x.rfind("<w:sectPr")            # final sectPr closes the body
head, tail = x[:head_end], x[tail_start:]

esc = lambda t: html.escape(t, quote=False)

def runs(t):
    out, last = [], 0
    for m in re.finditer(r'(\*\*[^*]+\*\*|`[^`]+`)', t):
        if m.start() > last: out.append(f'<w:r><w:t xml:space="preserve">{esc(t[last:m.start()])}</w:t></w:r>')
        s = m.group(0)
        if s.startswith("**"):
            out.append(f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{esc(s[2:-2])}</w:t></w:r>')
        else:
            out.append(f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr>'
                       f'<w:t xml:space="preserve">{esc(s[1:-1])}</w:t></w:r>')
        last = m.end()
    if last < len(t): out.append(f'<w:r><w:t xml:space="preserve">{esc(t[last:])}</w:t></w:r>')
    return "".join(out) or f'<w:r><w:t xml:space="preserve">{esc(t)}</w:t></w:r>'

IMAGES = []   # (relId, abspath, ext)

def png_size(path):
    with open(path,"rb") as f:
        f.read(16)
        w,h = struct.unpack(">II", f.read(8))
    return w,h

def figure(path, caption, number):
    """Inline image plus a numbered caption paragraph."""
    if not os.path.exists(path):
        return para(f"[MISSING FIGURE: {path}]", "BodyText")
    rid = f"rIdFig{len(IMAGES)+1}"
    IMAGES.append((rid, os.path.abspath(path), os.path.splitext(path)[1].lstrip(".")))
    pw, ph = png_size(path)
    cx = 5486400                      # 6 inches
    cy = int(cx * ph / pw)
    drawing = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
               f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{len(IMAGES)}" name="Figure {number}"/>'
               f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
               f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:nvPicPr><pic:cNvPr id="{len(IMAGES)}" name="Figure {number}"/><pic:cNvPicPr/></pic:nvPicPr>'
               f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
               f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
               f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
               f'</a:graphicData></a:graphic></wp:inline></w:drawing>')
    img = f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="180" w:after="60"/></w:pPr><w:r>{drawing}</w:r></w:p>'
    cap = (f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="220"/></w:pPr>'
           f'<w:r><w:rPr><w:b/><w:sz w:val="18"/></w:rPr><w:t xml:space="preserve">Figure {number}: </w:t></w:r>'
           f'<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t xml:space="preserve">{esc(caption)}</w:t></w:r></w:p>')
    return img + cap

def para(t, style=None, before=False):
    pr = "<w:pPr>"
    if style: pr += f'<w:pStyle w:val="{style}"/>'
    if before: pr += '<w:r><w:br w:type="page"/></w:r>'
    pr += "</w:pPr>"
    if before:
        return (f'<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
                f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{runs(t)}</w:p>')
    return f'<w:p>{pr}{runs(t)}</w:p>'

def table(rows):
    nc = max(len(r) for r in rows); w = 9020 // nc
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for _ in range(nc))
    body = ""
    for ri, r in enumerate(rows):
        cells = ""
        for ci in range(nc):
            txt = r[ci] if ci < len(r) else ""
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="EDEDED"/>' if ri == 0 else ""
            bold = "<w:b/>" if ri == 0 else ""
            cells += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}</w:tcPr>'
                      f'<w:p><w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>'
                      f'<w:r><w:rPr>{bold}<w:sz w:val="17"/></w:rPr>'
                      f'<w:t xml:space="preserve">{esc(txt)}</w:t></w:r></w:p></w:tc>')
        body += f"<w:tr>{cells}</w:tr>"
    return (f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
            f'<w:tblW w:w="9020" w:type="dxa"/></w:tblPr>'
            f'<w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl><w:p/>')

FIGN = [0]
md = open(md_file, encoding="utf8").read().split("\n")
out, i, first = [], 0, True
while i < len(md):
    L = md[i]
    if re.match(r'^\|', L) and i + 1 < len(md) and re.match(r'^\|[-| :]+\|$', md[i+1]):
        rows, j = [], i
        while j < len(md) and md[j].startswith("|"):
            if not re.match(r'^\|[-| :]+\|$', md[j]):
                rows.append([c.strip() for c in md[j].split("|")[1:-1]])
            j += 1
        out.append(table(rows)); i = j; continue
    m_img = re.match(r'^!\[(.*?)\]\((.*?)\)\s*$', L.strip())
    if m_img:
        FIGN[0] += 1
        out.append(figure(os.path.join(os.path.dirname(os.path.abspath(md_file)), m_img.group(2)),
                          m_img.group(1), FIGN[0]))
        i += 1; continue
    if L.startswith("### "):
        out.append(f'<w:p><w:pPr><w:spacing w:before="180" w:after="60"/></w:pPr>'
                   f'<w:r><w:rPr><w:b/><w:i/></w:rPr><w:t xml:space="preserve">{esc(L[4:])}</w:t></w:r></w:p>')
        i += 1; continue
    if L.startswith("## "):  out.append(para(L[3:], "Heading3")); i += 1; continue
    if L.startswith("# "):
        out.append(para(L[2:], "Heading2", before=not first)); first = False; i += 1; continue
    if re.match(r'^\s*[-*]\s+', L):
        out.append(f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/>'
                   f'<w:numId w:val="1"/></w:numPr></w:pPr>{runs(re.sub(r"^\s*[-*]\s+","",L))}</w:p>')
        i += 1; continue
    if re.match(r'^\s*\d+\.\s+', L):
        out.append(f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/>'
                   f'<w:numId w:val="2"/></w:numPr></w:pPr>{runs(re.sub(r"^\s*\d+\.\s+","",L))}</w:p>')
        i += 1; continue
    if L.strip() == "": i += 1; continue
    if L.strip().startswith("$$"):
        out.append(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:i/></w:rPr>'
                   f'<w:t xml:space="preserve">{esc(L.replace("$$","").strip())}</w:t></w:r></w:p>')
        i += 1; continue
    buf = [L]; i += 1
    while i < len(md) and md[i].strip() and not re.match(r'^[#|]', md[i]) \
          and not re.match(r'^\s*([-*]|\d+\.)\s', md[i]) and not md[i].strip().startswith("$$"):
        buf.append(md[i]); i += 1
    out.append(para(" ".join(buf).strip()))

new = head + "".join(out) + tail
open(doc, "w", encoding="utf8").write(new)

# retitle the cover sheet
x2 = open(doc, encoding="utf8").read()
for old, rep in (("THE TITLE OF THE PROJECT", title.upper()),
                 ("Your Name", "Kaushik Karthigeyan Senthilathiban"),
                 ("2020", "2026")):
    x2 = x2.replace(f"<w:t>{old}</w:t>", f"<w:t>{esc(rep)}</w:t>", 1)
    x2 = x2.replace(f'<w:t xml:space="preserve">{old}</w:t>', f'<w:t xml:space="preserve">{esc(rep)}</w:t>', 1)
open(doc, "w", encoding="utf8").write(x2)

# embed images: copy into word/media and register relationships
if IMAGES:
    media = os.path.join(work, "word", "media"); os.makedirs(media, exist_ok=True)
    rels_path = os.path.join(work, "word", "_rels", "document.xml.rels")
    rels = open(rels_path, encoding="utf8").read()
    add = ""
    for rid, src, ext in IMAGES:
        name = f"{rid}.{ext}"
        shutil.copyfile(src, os.path.join(media, name))
        add += (f'<Relationship Id="{rid}" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="media/{name}"/>')
    rels = rels.replace("</Relationships>", add + "</Relationships>")
    open(rels_path, "w", encoding="utf8").write(rels)
    ct_path = os.path.join(work, "[Content_Types].xml")
    ct = open(ct_path, encoding="utf8").read()
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Types ", '<Types ', 1)
        ct = ct.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
        open(ct_path, "w", encoding="utf8").write(ct)
    print(f"  embedded {len(IMAGES)} figures")

if os.path.exists(out_file): os.remove(out_file)
zf = zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED)
for root, _, files in os.walk(work):
    for f in files:
        p = os.path.join(root, f)
        zf.write(p, os.path.relpath(p, work))
zf.close()
print(f"  wrote {out_file} ({os.path.getsize(out_file)//1024} KB, {len(out)} blocks spliced)")
