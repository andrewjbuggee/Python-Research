#!/usr/bin/env python3
"""Build build_review_v4.js from build_review_v3.js:
   - insert the new Section 7 (section7.js) and renumber old Sections 7-11 to 8-12
   - fix every cross-reference, update the title line, abstract, 1.2, 1.3 and the reading order
   - typeset subscripts/superscripts (shared rich-text helper)
   - add the new references
"""
import re

src = open("build_review_v3.js", encoding="utf-8").read()
s = src

def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, f"expected {count}, found {n}: {old[:90]!r}"
    s = s.replace(old, new)

# ------------------------------------------------------------------ helpers: rich text
rep('''const P = (t, opts = {}) => {
  // supports *italic* segments via simple split on '*'
  const parts = t.split("*");
  const runs = parts.map((s, i) => new TextRun({ text: s, italics: i % 2 === 1, font: FONT, size: BODY }));
  return new Paragraph({ children: runs, spacing: { after: 160, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...opts });
};''',
'''// Rich text: *italic* toggles (when stars is true), _{sub} and ^{sup}; bare tokens such as N_d, r_e, S_LWP
// are typeset with a true subscript.
const SUBRE = /(?:(?<![A-Za-z0-9_])(ACI|SO|[A-Za-z])|([τασκβρξχωλ]))_([A-Za-z0-9∗]{1,4})(?![A-Za-z0-9_{])/g;
const rich = (t, o = {}) => {
  const size = o.size || BODY, stars = o.stars !== false;
  t = t.replace(SUBRE, (mm, a, b, c) => (a || b) + "_{" + c + "}");
  const out = []; let it = false, buf = "";
  const mk = (text, extra = {}) => new TextRun({ text, italics: it || !!o.italics, bold: !!o.bold, font: FONT, size, ...extra });
  const flush = () => { if (buf) { out.push(mk(buf)); buf = ""; } };
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (stars && c === "*") { flush(); it = !it; continue; }
    if ((c === "_" || c === "^") && t[i + 1] === "{") {
      const j = t.indexOf("}", i + 2);
      if (j > 0) { flush(); out.push(mk(t.slice(i + 2, j), c === "_" ? { subScript: true } : { superScript: true })); i = j; continue; }
    }
    buf += c;
  }
  flush();
  return out.length ? out : [mk("")];
};
const P = (t, opts = {}) => new Paragraph({ children: rich(t), spacing: { after: 160, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...opts });''')

rep('''const Bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun({ text: t, font: FONT, size: BODY })], spacing: { after: 80, line: 276 } });''',
    '''const Bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: rich(t, { stars: false }), spacing: { after: 80, line: 276 } });''')
rep('''const PB = (lead, t) => { const parts = t.split("*"); const runs = [new TextRun({ text: lead, bold: true, font: FONT, size: BODY })].concat(parts.map((s, i) => new TextRun({ text: s, italics: i % 2 === 1, font: FONT, size: BODY }))); return new Paragraph({ children: runs, spacing: { after: 160, line: 276 }, alignment: AlignmentType.JUSTIFIED }); };''',
    '''const PB = (lead, t) => new Paragraph({ children: rich(lead, { bold: true, stars: false }).concat(rich(t)), spacing: { after: 160, line: 276 }, alignment: AlignmentType.JUSTIFIED });''')
rep('''const Num = (t, ref = "nums") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: [new TextRun({ text: t, font: FONT, size: BODY })], spacing: { after: 80, line: 276 } });''',
    '''const Num = (t, ref = "nums") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: rich(t, { stars: false }), spacing: { after: 80, line: 276 } });''')
rep('''  children: [new Paragraph({ children: [new TextRun({ text: t, bold, font: FONT, size: 20 })], spacing: { after: 0, line: 260 } })]''',
    '''  children: [new Paragraph({ children: rich(t, { bold, size: 20, stars: false }), spacing: { after: 0, line: 260 } })]''')

# the one caret exponent in the old text
rep("optical thickness scales as N_d^(1/3)", "optical thickness scales as N_d^{1/3}")

# headings keep with next
rep("paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } }", "paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } }")
rep("paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } }", "paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } }")

# ------------------------------------------------------------------ title line
rep("9 November 2026 (revision 3: adds Section 11 on questions addressable within the EPCAPE turbulence–brightness project; revision 2 added the Hobbs, 1993, audit)",
    "10 November 2026 (revision 4: adds Section 7 on the physics behind the results, with equations and figures, and renumbers the later sections; revision 3 added the section on questions addressable within the EPCAPE turbulence–brightness project, now Section 12; revision 2 added the Hobbs, 1993, audit)")

# ------------------------------------------------------------------ abstract
rep("and new ground-based campaigns such as EPCAPE are being used to close the remaining gaps. A final section singles out the surface-radiation perspective,",
    "and new ground-based campaigns such as EPCAPE are being used to close the remaining gaps. A physics section then steps through the mechanisms behind these results (activation, condensational growth, coalescence, extinction, multiple scattering, absorption and entrainment) with equations, worked numbers and figures, drawing on two textbooks and three sets of course and reference material. A further section singles out the surface-radiation perspective,")

# ------------------------------------------------------------------ 1.2 sources
rep("Those statements are used in Sections 3.5 and 9.2 as a benchmark:", "Those statements are used in Sections 3.5 and 10.2 as a benchmark:")
anchor_12 = 'children.push(H("1.3 Organization and key terms", HeadingLevel.HEADING_2));'
rep(anchor_12,
    'children.push(P("A fourth group of sources supports the physics section (Section 7): the textbooks of Rogers and Yau (1989) and Bohren and Clothiaux (2006), two lecture decks from an MIT course (Ardon-Dryer and Garimella, 2014a, 2014b), the encyclopedia article of Li et al. (2017), and Robert Wood\'s notes on seminal papers from his University of Washington course on aerosol–cloud–climate interactions (Wood, 2015). These were supplied by the reader and are treated as authoritative for the physics. Figures redrawn from them were recomputed from the source equations and checked against the values tabulated in the originals. Wood\'s notes are informal course material and are cited as such; where they summarize a paper, the paper itself is the primary source."));\n'
    + anchor_12)

# ------------------------------------------------------------------ 1.3 organization
rep("Sections 2–6 proceed chronologically. Section 7 steps outside the chronology to treat surface radiation as a theme, because it cuts across all eras and because it is the perspective most relevant to EPCAPE. Section 8 synthesizes, Section 9 lists open questions (Section 9.1 from the recent literature, Section 9.2 from the 1993 audit), Section 10 gives a reading order, and Section 11 identifies which of the open questions can be addressed within the EPCAPE turbulence–brightness project.",
    "Sections 2–6 proceed chronologically. Section 7 steps through the physics and chemistry behind the results of those sections, with equations and figures, and can be read before them or alongside them. Section 8 steps outside the chronology to treat surface radiation as a theme, because it cuts across all eras and because it is the perspective most relevant to EPCAPE. Section 9 synthesizes, Section 10 lists open questions (Section 10.1 from the recent literature, Section 10.2 from the 1993 audit), Section 11 gives a reading order, and Section 12 identifies which of the open questions can be addressed within the EPCAPE turbulence–brightness project.")

# ------------------------------------------------------------------ 3.5
rep("Section 9.2 returns to each of these problems", "Section 10.2 returns to each of these problems")

# ------------------------------------------------------------------ insert Section 7 and renumber old 7
rep("// 7. Surface radiation theme\n",
    '// 7. Physics (new in revision 4)\nrequire("./section7.js")({ H, rich, cell, tw: 9360, FONT, BODY }).forEach((x) => children.push(x));\n\n// 8. Surface radiation theme\n')
rep('H("7. A cross-cutting theme: the surface radiation perspective"', 'H("8. A cross-cutting theme: the surface radiation perspective"')
rep("using the Feingold et al. (2022) form.", "using the Feingold et al. (2022) form (Sections 7.8 and 7.10 give the algebra).")

# ------------------------------------------------------------------ old 8 -> 9, 9 -> 10
rep('H("8. Where the field stands"', 'H("9. Where the field stands"')
rep('H("9. Open questions"', 'H("10. Open questions"')
rep('H("9.1 Questions arising from the 2017–2026 literature"', 'H("10.1 Questions arising from the 2017–2026 literature"')
rep('H("9.2 Outstanding problems posed in 1993 and their status in 2026"', 'H("10.2 Outstanding problems posed in 1993 and their status in 2026"')
rep("continuing the list of Section 9.1,", "continuing the list of Section 10.1,")
rep("is the question that Sections 7 and 9.1 (questions 8 and 9) pose.", "is the question that Sections 8 and 10.1 (questions 8 and 9) pose.")

# ------------------------------------------------------------------ old 10 -> 11 (reading order)
rep('H("10. Suggested reading order"', 'H("11. Suggested reading order"')
rep('children.push(Num("Twomey (1977) — the four-page original. Then Oreopoulos and Platnick (2006) for the susceptibility algebra.", "roadmap"));',
    'children.push(Num("Section 7 of this review, with Rogers and Yau (1989, Chapters 6–8) and Bohren and Clothiaux (2006, Chapter 5) at hand — the physics, read first or alongside the papers. Wood\'s (2015) notes are a companion to items 2, 3, 7 and 8 below.", "roadmap"));\n'
    'children.push(Num("Twomey (1977) — the four-page original. Then Oreopoulos and Platnick (2006) for the susceptibility algebra.", "roadmap"));')

# ------------------------------------------------------------------ old 11 -> 12
rep('H("11. Open questions addressable within the EPCAPE turbulence–brightness project"', 'H("12. Open questions addressable within the EPCAPE turbulence–brightness project"')
for k, name in [("1", "The proposal and how its work is divided"), ("2", "Selection criteria"), ("3", "Candidate questions"),
                ("4", "The boundary with the turbulence work"), ("5", "Summary and recommended sequence")]:
    rep(f'H("11.{k} {name}"', f'H("12.{k} {name}"')
rep("This section asks which of the open questions in Section 9 can be answered", "This section asks which of the open questions in Section 10 can be answered")
rep("it is open in the literature reviewed in Sections 8–9;", "it is open in the literature reviewed in Sections 9–10;")
rep("It addresses questions 5 and 11 of Section 9.", "It addresses questions 5 and 11 of Section 10.")
rep("It addresses questions 7 and 18 of Section 9.", "It addresses questions 7 and 18 of Section 10.")
rep("This is question 8 of Section 9, restated for the project.", "This is question 8 of Section 10, restated for the project.")
rep("Section 7 notes that the transmission-susceptibility expression used there to estimate its size", "Section 8 notes that the transmission-susceptibility expression used there to estimate its size")
rep("It addresses questions 1, 11 and 17 of Section 9.", "It addresses questions 1, 11 and 17 of Section 10.")
rep('"Open question (Section 9)"', '"Open question (Section 10)"')
rep('"Dispersion effect (Sec. 11.3)"', '"Dispersion effect (Secs. 7.7 and 12.3)"')

# ------------------------------------------------------------------ references
new_refs = {
    '"Arola, A.,': '"Ardon-Dryer, K., & Garimella, S. (2014a). Aerosols and clouds [Lecture slides: Section 3, Lecture 1, 6 November 2014]. 12.335/12.835 Experimental Atmospheric Chemistry, Fall 2014. MIT OpenCourseWare, Massachusetts Institute of Technology. http://ocw.mit.edu [folder]",\n'
                   '"Ardon-Dryer, K., & Garimella, S. (2014b). [Lecture slides: Section 3, Lecture 2, 18 November 2014; saturation, Köhler theory, CCN, κ-Köhler theory and collision–coalescence]. 12.335/12.835 Experimental Atmospheric Chemistry, Fall 2014. MIT OpenCourseWare, Massachusetts Institute of Technology. http://ocw.mit.edu [folder]",\n',
    '"Bréon, F.-M.,': '"Bohren, C. F., & Clothiaux, E. E. (2006). Fundamentals of Atmospheric Radiation: An Introduction with 400 Problems. Wiley-VCH, Weinheim. ISBN 3-527-40503-8. [folder]",\n',
    '"Lim, K.-S. S.,': '"Li, Z., Rosenfeld, D., & Fan, J. (2017). Aerosols and their impact on radiation, clouds, precipitation, and severe weather events. Oxford Research Encyclopedia of Environmental Science. https://doi.org/10.1093/acrefore/9780199389414.013.126 [folder]",\n',
    '"Rosenfeld, D., Zhu, Y.,': '"Rogers, R. R., & Yau, M. K. (1989). A Short Course in Cloud Physics (3rd ed.). Pergamon Press, Oxford. [folder; Chapters 1–10]",\n',
    '"Yeom, J. M.,': '"Wood, R. (2015). ATM S 591: Aerosol–cloud–climate interactions, Spring 2015 [course schedule, reading list and notes on seminal papers, including Twomey (1977), Albrecht (1989), Jones et al. (1994), Ackerman et al. (2004), McFiggans et al. (2006), Bretherton et al. (2007) and Stevens and Feingold (2009)]. University of Washington. http://faculty.washington.edu/robwood2/wordpress/?page_id=535",\n',
}
for anchor, text in new_refs.items():
    n = s.count("\n" + anchor)
    assert n == 1, (anchor, n)
    s = s.replace("\n" + anchor, "\n" + text + anchor)

# ------------------------------------------------------------------ output name
rep('fs.writeFileSync("ACI_Radiative_Impacts_Literature_Review_v3.docx", buf);', 'fs.writeFileSync("ACI_Radiative_Impacts_Literature_Review_v4_raw.docx", buf);')

open("build_review_v4.js", "w", encoding="utf-8").write(s)

# ------------------------------------------------------------------ report: leftover references to old numbering
print("built build_review_v4.js")
for pat in [r"Sections? 9\.[12]", r"Sections? 11\b", r"Sec\. 11", r"Section 9\b", r"Sections 8–9", r"Section 7 notes", r"Sections 7 and"]:
    hits = [m.start() for m in re.finditer(pat, s)]
    print(f"  leftover {pat!r}: {len(hits)}")
# tokens that the subscript rule will convert in the pre-existing text
SUB = re.compile(r"(?:(?<![A-Za-z0-9_])(ACI|SO|[A-Za-z])|([τασκβρξχωλ]))_([A-Za-z0-9∗]{1,4})(?![A-Za-z0-9_{])")
body = s[s.index("// ---------- content ----------"): s.index("const refs = [")]
from collections import Counter
c = Counter(m.group(0) for m in SUB.finditer(body))
print("  subscript tokens in body text:", dict(c))
unders = Counter(re.findall(r"\S*_\S*", re.sub(SUB, "", body)))
print("  other underscores in body:", dict(list(unders.items())[:20]))
