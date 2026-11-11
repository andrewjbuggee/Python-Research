#!/bin/bash
# build -> finalize -> pdf -> page images -> 2-up sheets
set -e
cd "$(dirname "$0")"
node build_review_v4.js >/dev/null
python3 finalize_v4.py
python3 ${DOCX_SCRIPTS:-/mnt/skills/public/docx/scripts/office}/validate.py ACI_Radiative_Impacts_Literature_Review_v4.docx 2>&1 | tail -1
rm -f ACI_Radiative_Impacts_Literature_Review_v4.pdf
python3 ${DOCX_SCRIPTS:-/mnt/skills/public/docx/scripts/office}/soffice.py --headless --convert-to pdf ACI_Radiative_Impacts_Literature_Review_v4.docx >/dev/null 2>&1
pdfinfo ACI_Radiative_Impacts_Literature_Review_v4.pdf | grep Pages
mkdir -p v4pages; rm -f v4pages/*; pdftoppm -jpeg -r 100 ACI_Radiative_Impacts_Literature_Review_v4.pdf v4pages/p >/dev/null 2>&1
python3 - <<'PY'
from PIL import Image
import glob
files=sorted(glob.glob("v4pages/p-*.jpg"))
def sheet(a,b,name):
    ims=[Image.open(files[i-1]) for i in (a,b) if i<=len(files)]
    W=sum(i.width for i in ims)+10*(len(ims)-1); H=max(i.height for i in ims)
    s=Image.new("RGB",(W,H),"white"); x=0
    for im in ims: s.paste(im,(x,0)); x+=im.width+10
    s.save("v4pages/"+name,quality=85)
for a in range(1,len(files)+1,2): sheet(a,a+1,f"s{a:02d}.jpg")
print(len(files),"pages")
PY
