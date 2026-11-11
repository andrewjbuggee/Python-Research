#!/usr/bin/env python3
"""Post-process the raw v4 docx: make stand-alone equations display equations (m:oMathPara),
drop ZIP directory entries, and write the final file."""
import re, zipfile, sys
src, dst = "ACI_Radiative_Impacts_Literature_Review_v4_raw.docx", "ACI_Radiative_Impacts_Literature_Review_v4.docx"
zin = zipfile.ZipFile(src)
names = [i for i in zin.infolist() if not i.filename.endswith("/")]
doc = zin.read("word/document.xml").decode("utf-8")
assert 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"' in doc
count = 0
def fix(mm):
    global count
    p = mm.group(0)
    if "<m:oMath>" in p and "<w:r>" not in p and "<w:r " not in p and "<m:oMathPara>" not in p:
        count += 1
        p = p.replace("<m:oMath>", '<m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr><m:oMath>', 1)
        i = p.rfind("</m:oMath>")
        p = p[:i] + "</m:oMath></m:oMathPara>" + p[i + len("</m:oMath>"):]
    return p
doc2 = re.sub(r"<w:p>(?:(?!</w:p>).)*</w:p>", fix, doc, flags=re.S)
print("display equations wrapped:", count, "| total oMath:", doc.count("<m:oMath>"))
zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
for i in names:
    data = doc2.encode("utf-8") if i.filename == "word/document.xml" else zin.read(i.filename)
    zout.writestr(i.filename, data)
zout.close()
print("wrote", dst)
