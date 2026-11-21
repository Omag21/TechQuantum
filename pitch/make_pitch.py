"""Build the 3-minute TechQuantum pitch (pitch/TechQuantum_Pitch.pptx) from the project figures.

    python pitch/make_pitch.py        (needs python-pptx and Pillow)
"""
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'outputs/figures'
OUT = ROOT / 'TechQuantum_Pitch.pptx'
import tempfile
TMP = Path(tempfile.mkdtemp(prefix='techquantum_pitch_'))

NIGHT, NIGHT2 = RGBColor(0x0F, 0x1B, 0x2D), RGBColor(0x1B, 0x2E, 0x4A)
INK, SOFT = RGBColor(0x14, 0x20, 0x2E), RGBColor(0x4A, 0x55, 0x63)
WHITE, PAPER = RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xF3, 0xF6, 0xFA)
BLUE, ICE = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x8F, 0xBC, 0xF0)
ORANGE, LINE = RGBColor(0xEB, 0x68, 0x34), RGBColor(0xD9, 0xE0, 0xE8)
HEAD, BODY = 'Cambria', 'Calibri'

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, spacing=None):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing: p.space_after = Pt(spacing)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run(); r.text = t
            r.font.size = Pt(o.get('size', size)); r.font.name = o.get('font', font)
            r.font.bold = o.get('bold', bold); r.font.italic = o.get('italic', False)
            r.font.color.rgb = o.get('color', color)
    return tb


def box(slide, x, y, w, h, fill, line=None, radius=0.08, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line: s.line.color.rgb = line; s.line.width = Pt(1.25)
    else: s.line.fill.background()
    s.shadow.inherit = False
    if name: s.name = name
    return s


def picture(slide, path, x, y, w=None, h=None, name=None, alt=''):
    iw, ih = Image.open(path).size
    if w and not h: h = w * ih / iw
    if h and not w: w = h * iw / ih
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    if name: p.name = name
    p._element.nvPicPr.cNvPr.set('descr', alt)
    return p


def title(slide, t, color=INK):
    text(slide, 0.6, 0.45, 12.1, 0.9, t, size=36, color=color, font=HEAD, bold=True, name='Title')


def footer(slide, n, dark=False):
    c = ICE if dark else SOFT
    text(slide, 0.6, 7.0, 9, 0.3, 'TechQuantum · Qiskit Fall Fest 2026 · EMSI Casablanca × Quantum Africa', size=10, color=c)
    text(slide, 11.7, 7.0, 1.0, 0.3, f'{n} / 4', size=10, color=c, align=PP_ALIGN.RIGHT)


# Logos (converted to PNG; they belong to their owners and are used for the event pitch only).
logo_emsi = TMP / 'emsi.png'
Image.open(ROOT / 'assets/emsi-logo.webp').convert('RGBA').save(logo_emsi)
logo_qa = ROOT / 'assets/quantum-africa-logo.png'

# ------------------------------------------------------------------ 1. cover
s = prs.slides.add_slide(BLANK); bg(s, NIGHT)
box(s, 0.6, 0.55, 3.5, 0.75, WHITE, radius=0.2, name='Logo card')
picture(s, logo_emsi, 0.8, 0.65, h=0.55, name='EMSI logo', alt='Logo EMSI')
picture(s, logo_qa, 2.55, 0.65, h=0.55, name='Quantum Africa logo', alt='Logo Quantum Africa')
text(s, 0.6, 1.75, 11, 0.4, 'QISKIT FALL FEST 2026 · QUANTUM CHALLENGE', size=14, color=ICE, bold=True)
text(s, 0.6, 2.25, 11.6, 1.9, 'Un noyau quantique à 8 qubits pour reconnaître les cultures de Doukkala',
     size=44, color=WHITE, font=HEAD, bold=True, name='Title')
text(s, 0.6, 4.2, 10.5, 0.9, 'Luzerne, maïs ou blé à partir de 13 mesures Sentinel-2, comparé équitablement '
     'aux meilleures méthodes classiques', size=18, color=RGBColor(0xC9, 0xD3, 0xE0))
text(s, 0.6, 5.35, 4, 0.35, 'ÉQUIPE TECHQUANTUM', size=12, color=ORANGE, bold=True)
for i, n in enumerate(['Gédéon AKOUBOUA NGOKELELE', 'Taha EL MOUHADDEB', 'Kawtar KESLAK']):
    box(s, 0.6 + i * 4.1, 5.8, 3.85, 0.7, NIGHT2, radius=0.2, name=f'Member {i + 1}')
    text(s, 0.6 + i * 4.1, 5.8, 3.85, 0.7, n, size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 7.0, 9, 0.3, 'EMSI Casablanca · Laboratoire LPRI × Quantum Africa · 21 novembre 2026', size=10, color=ICE)
s.notes_slide.notes_text_frame.text = (
    "[≈ 25 s] Bonjour, nous sommes l'équipe TechQuantum : Gédéon, Taha et Kawtar. Notre défi : reconnaître la culture "
    "d'une parcelle de Doukkala, luzerne, maïs ou blé, à partir de 13 mesures du satellite Sentinel-2. Nous avons "
    "remplacé le petit réseau quantique fourni par un noyau quantique à 8 qubits, et surtout nous l'avons comparé "
    "honnêtement aux méthodes classiques.")

# ------------------------------------------------------------------ 2. method
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, 'Une parcelle devient un état quantique à 8 qubits')
steps = [('1', '13 mesures', 'Bandes B1 à B12 + NDVI'), ('2', 'PCA → 8 variables', 'Ajustée sur l’entraînement seul'),
         ('3', 'Circuit ZZ', '8 qubits, échelle 0,04, aucun angle entraîné'),
         ('4', 'Noyau de fidélité', 'K = |⟨ψ(x)|ψ(x′)⟩|², simulation exacte'), ('5', 'SVM', 'Luzerne, maïs ou blé')]
cw, gap, y0 = 2.22, 0.25, 1.55
for i, (n, t, d) in enumerate(steps):
    x = 0.6 + i * (cw + gap); quantum = i in (2, 3)
    box(s, x, y0, cw, 1.75, PAPER if not quantum else RGBColor(0xE6, 0xF0, 0xFC), line=BLUE if quantum else LINE,
        name=f'Step {n}')
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.2), Inches(y0 + 0.2), Inches(0.45), Inches(0.45))
    c.fill.solid(); c.fill.fore_color.rgb = BLUE if quantum else INK; c.line.fill.background(); c.name = f'Step {n} badge'
    c.text_frame.text = n; p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    p.runs[0].font.size = Pt(14); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE; p.runs[0].font.name = BODY
    c.text_frame.margin_left = c.text_frame.margin_right = c.text_frame.margin_top = c.text_frame.margin_bottom = 0
    text(s, x + 0.2, y0 + 0.78, cw - 0.4, 0.4, t, size=16, bold=True)
    text(s, x + 0.2, y0 + 1.13, cw - 0.4, 0.55, d, size=12, color=SOFT)
    if i < 4:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + cw + 0.04), Inches(y0 + 0.78), Inches(0.17), Inches(0.22))
        a.fill.solid(); a.fill.fore_color.rgb = LINE; a.line.fill.background(); a.name = f'Arrow {n}'
box(s, 0.6, 3.6, 12.13, 2.05, PAPER, name='Circuit card')
cw_img = 1.8 * Image.open(FIG / 'fig0_circuit.png').size[0] / Image.open(FIG / 'fig0_circuit.png').size[1]
picture(s, FIG / 'fig0_circuit.png', 0.6 + (12.13 - cw_img) / 2, 3.72, h=1.8, name='Circuit', alt='Une répétition du circuit ZZ à 8 qubits : portes H, rotations de phase et CNOT en chaîne')
box(s, 0.6, 5.9, 12.13, 0.85, RGBColor(0xFD, 0xEE, 0xE7), radius=0.15, name='Fair comparison')
text(s, 0.85, 5.9, 11.7, 0.85, [[('Comparaison équitable : ', {'bold': True, 'color': ORANGE}),
     ('on ne change que le noyau. Mêmes 8 variables, même SVM, mêmes parcelles et 25 réglages chacun, choisis par '
      'validation croisée sur l’entraînement.', {})]], size=15, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 2)
s.notes_slide.notes_text_frame.text = (
    "[≈ 45 s] On résume les 13 bandes en 8 variables par PCA, une par qubit. Chaque parcelle est encodée dans un "
    "circuit ZZ, puis la similarité entre deux parcelles est le recouvrement de leurs états quantiques. Un SVM classique "
    "classe avec ces similarités. Point clé : on ne change que le noyau, avec les mêmes entrées, le même classifieur et "
    "le même budget de réglage que les contrôles classiques. Notre noyau est identique à celui de Qiskit à 10^-14 près.")

# ------------------------------------------------------------------ 3. results
s = prs.slides.add_slide(BLANK); bg(s, WHITE)
title(s, 'Au niveau de la référence, sans avantage quantique')
text(s, 0.6, 1.6, 4.3, 1.4, '0,886', size=88, color=BLUE, font=HEAD, bold=True, name='Big number')
text(s, 0.6, 2.95, 4.3, 0.4, 'macro-F1 sur 150 parcelles de validation', size=14, color=SOFT)
rows = [('>', 'QNN de départ', '0,856'), ('≈', 'SVM de référence', '0,880'), ('<', 'SVM RBF / linéaire', '0,906 / 0,913')]
for i, (sign, lab, val) in enumerate(rows):
    y = 3.6 + i * 0.62
    box(s, 0.6, y, 4.3, 0.5, PAPER, radius=0.2, name=f'Compare {i + 1}')
    text(s, 0.8, y, 0.4, 0.5, sign, size=20, color=ORANGE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 1.2, y, 2.0, 0.5, lab, size=15, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 3.1, y, 1.65, 0.5, val, size=15, bold=True, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 5.55, 4.3, 1.1, 'Écarts non significatifs : les intervalles à 95 % du bootstrap spatial contiennent 0. '
     'Déterministe : graines 0, 1, 2 identiques.', size=13, color=SOFT)
box(s, 5.3, 2.5, 7.43, 3.05, PAPER, name='Chart card')
picture(s, FIG / 'fig2_validation_scores.png', 5.5, 2.7, w=7.03, name='Validation chart',
        alt='Macro-F1 de validation : noyau quantique 0,886, RBF 0,906, linéaire 0,913, référence 0,880, QNN de départ 0,856')
footer(s, 3)
s.notes_slide.notes_text_frame.text = (
    "[≈ 45 s] Notre noyau quantique atteint 0,886 de macro-F1 : mieux que le QNN fourni, au niveau du SVM de référence. "
    "Les SVM classiques sur les mêmes 8 variables font un peu mieux, mais le bootstrap par blocs spatiaux montre que ces "
    "écarts ne sont pas significatifs. Nous ne revendiquons donc aucun avantage quantique. Tous les modèles testés, "
    "même les plus puissants, plafonnent vers 0,92 : la limite vient des données, une seule date d'image.")

# ------------------------------------------------------------------ 4. take-aways
s = prs.slides.add_slide(BLANK); bg(s, NIGHT)
title(s, 'Ce que nous retenons', color=WHITE)
box(s, 0.6, 1.55, 7.6, 3.2, WHITE, name='Concentration card')
picture(s, FIG / 'fig1_kernel_bandwidth.png', 0.75, 1.75, w=7.3, name='Kernel concentration',
        alt='La similarité entre parcelles chute vers 1/256 quand l’échelle augmente et la macro-F1 de validation croisée passe de 0,91 à 0,36')
text(s, 0.6, 4.95, 7.6, 1.5, [[('Le piège de la concentration. ', {'bold': True, 'color': ICE}),
     ('Au-delà d’une échelle de 0,2, toutes les parcelles semblent orthogonales (similarité ≈ 1/256) et la macro-F1 '
      'chute de 0,91 à 0,36. Une petite échelle (0,04) est indispensable.', {'color': WHITE})]], size=15)
cards = [('15 s', 'par graine, contre 69 min pour le QNN de départ sur le même PC'),
         ('+0,045', 'de 228 à 457 parcelles d’entraînement, puis un plateau'),
         ('1 commande', 'regénère tout : python -m experiments.predict_test')]
for i, (big, small) in enumerate(cards):
    y = 1.55 + i * 1.12
    box(s, 8.55, y, 4.18, 0.97, NIGHT2, radius=0.12, name=f'Takeaway {i + 1}')
    text(s, 8.8, y + 0.1, 3.8, 0.45, big, size=24, color=ORANGE, font=HEAD, bold=True)
    text(s, 8.8, y + 0.52, 3.8, 0.42, small, size=12, color=RGBColor(0xDD, 0xE4, 0xEE))
text(s, 8.55, 5.0, 4.18, 1.4, 'Limites : une seule cohorte (Doukkala, une date), simulation sans bruit. '
     'Suite : inférence sur un vrai processeur IBM.', size=13, color=ICE)
text(s, 0.6, 6.45, 12.1, 0.45, 'Merci ! Questions ?', size=20, color=WHITE, font=HEAD, bold=True)
footer(s, 4, dark=True)
s.notes_slide.notes_text_frame.text = (
    "[≈ 55 s] Notre résultat le plus instructif : avec 8 qubits l'espace a 256 dimensions. Si on encode trop fort, deux "
    "parcelles différentes deviennent presque orthogonales et le modèle ne peut plus rien apprendre : la macro-F1 chute "
    "de 0,91 à 0,36. Il faut une petite échelle. Le modèle est aussi rapide, 15 secondes contre 69 minutes pour le QNN, "
    "il profite de données supplémentaires jusqu'à environ 450 parcelles, et tout se regénère avec une seule commande. "
    "Limites : une seule région et une date, simulation sans bruit. Merci, nous sommes prêts pour vos questions.")

prs.core_properties.title = 'TechQuantum — Pitch Quantum Challenge'
prs.core_properties.author = 'Équipe TechQuantum'
prs.save(OUT)
print('saved', OUT)
