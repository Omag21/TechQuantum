"""TechQuantum — interface de détection de culture par noyau quantique.

    .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.inference import BANDS, LABELS_FR, CropPredictor, read_image, rgb_preview, summarize_field
from challenge.submission import CONFIG, FEATURES

st.set_page_config(page_title='TechQuantum · Détection de cultures', page_icon='🛰️', layout='wide')

# Class colours follow the entity everywhere (validated categorical slots 1-3).
COLORS = {'Luzerne': '#1baf7a', 'Maïs': '#eb6834', 'Blé': '#2a78d6'}
ICONS = {'Luzerne': '🌿', 'Maïs': '🌽', 'Blé': '🌾'}

st.markdown("""
<style>
:root { --tq-ink:#14202e; --tq-soft:#5b6573; --tq-line:#e3e0d8; --tq-card:#ffffff; --tq-blue:#2a78d6; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
.tq-hero { background: linear-gradient(120deg,#0f1b2d 0%,#1b3355 100%); color:#eef2f7; border-radius:18px; padding:28px 32px; margin-bottom:18px; }
.tq-hero h1 { font-size: 2rem; margin:0 0 6px 0; color:#fff; }
.tq-hero p { margin:0; color:#c9d3e0; font-size:1.02rem; }
.tq-kicker { font-size:.8rem; letter-spacing:.12em; color:#8fbcf0; font-weight:700; margin-bottom:8px; }
.tq-result { border-radius:16px; padding:22px 24px; border:2px solid var(--tq-line); background:var(--tq-card); }
.tq-result .lbl { font-size:.85rem; color:var(--tq-soft); text-transform:uppercase; letter-spacing:.08em; }
.tq-result .crop { font-size:2.6rem; font-weight:800; line-height:1.1; margin:4px 0 6px 0; }
.tq-pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.82rem; font-weight:600; margin-right:6px; }
.tq-ok { background:#e3f4ec; color:#11643f; } .tq-warn { background:#fdf0dc; color:#7a4a00; }
.tq-note { font-size:.86rem; color:var(--tq-soft); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner='Entraînement du modèle quantique sur les 914 parcelles (≈ 15 s, une seule fois)…')
def get_predictor():
    return CropPredictor()


def result_card(row):
    crop = row['culture']; color = COLORS[crop]
    fam = ('<span class="tq-pill tq-ok">Parcelle proche des données d’entraînement</span>' if row['familier'] else
           '<span class="tq-pill tq-warn">⚠ Parcelle inhabituelle : prédiction à prendre avec prudence</span>')
    st.markdown(f"""
    <div class="tq-result" style="border-color:{color}">
      <div class="lbl">Culture détectée</div>
      <div class="crop" style="color:{color}">{ICONS[crop]} {crop}</div>
      {fam}
    </div>""", unsafe_allow_html=True)


def score_chart(row):
    d = pd.DataFrame({'Culture': [LABELS_FR[c] for c in ['alfalfa', 'maize', 'wheat']],
                      'Score': [row[f'score_{c}'] for c in ['alfalfa', 'maize', 'wheat']]})
    base = alt.Chart(d).encode(
        y=alt.Y('Culture:N', sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelFontSize=13)),
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%', grid=True), title='Score relatif'),
        tooltip=[alt.Tooltip('Culture:N'), alt.Tooltip('Score:Q', format='.1%')])
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color('Culture:N', scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=None))
    text = base.mark_text(align='left', dx=6, color='#14202e').encode(text=alt.Text('Score:Q', format='.0%'))
    st.altair_chart((bars + text).properties(height=160), use_container_width=True)
    st.caption('Scores SVM convertis en parts relatives (non calibrés comme des probabilités).')


def spectrum_chart(field, predictor):
    ref = predictor.class_spectra[BANDS].rename(index=LABELS_FR).reset_index().melt('target', var_name='Bande', value_name='Réflectance')
    ref = ref.rename(columns={'target': 'Série'})
    mine = pd.DataFrame({'Bande': BANDS, 'Réflectance': field[BANDS].iloc[0].values, 'Série': 'Votre parcelle'})
    d = pd.concat([ref, mine])
    domain = list(COLORS) + ['Votre parcelle']; rng = list(COLORS.values()) + ['#14202e']
    chart = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
        x=alt.X('Bande:N', sort=BANDS, title='Bande Sentinel-2'),
        y=alt.Y('Réflectance:Q', title='Réflectance (×10 000)'),
        color=alt.Color('Série:N', scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(orient='bottom', title=None, columns=2, labelLimit=200)),
        strokeDash=alt.condition(alt.datum.Série == 'Votre parcelle', alt.value([1, 0]), alt.value([5, 3])),
        tooltip=['Série', 'Bande', alt.Tooltip('Réflectance:Q', format=',.0f')])
    st.altair_chart(chart.properties(height=340), use_container_width=True)
    st.caption('Traits pointillés : spectre médian de chaque culture dans les données d’entraînement.')


def show_prediction(field, predictor, extra=None):
    row = predictor.predict(field).iloc[0]
    c1, c2 = st.columns([1, 1.25], gap='large')
    with c1:
        result_card(row)
        st.write('')
        score_chart(row)
        if extra: extra()
    with c2:
        st.markdown('##### Signature spectrale')
        spectrum_chart(field, predictor)
    with st.expander('Valeurs envoyées au modèle (13 variables)'):
        st.dataframe(field.round(3), hide_index=True, use_container_width=True)
    return row


# ---------------------------------------------------------------- header & sidebar
st.markdown("""
<div class="tq-hero">
  <div class="tq-kicker">QISKIT FALL FEST 2026 · EMSI × QUANTUM AFRICA · ÉQUIPE TECHQUANTUM</div>
  <h1>🛰️ Détection de cultures par noyau quantique</h1>
  <p>Déposez l’image Sentinel-2 d’une parcelle : un circuit quantique à 8 qubits l’encode, puis un SVM reconnaît
  la culture parmi <b>luzerne, maïs et blé</b>.</p>
</div>""", unsafe_allow_html=True)

predictor = get_predictor()

with st.sidebar:
    st.markdown('### 🧠 Le modèle')
    fm = CONFIG['feature_map']
    st.markdown(f"""
- **Encodage** : carte ZZ, {fm['qubits']} qubits, {fm['reps']} répétitions
- **Échelle** : {CONFIG['params']['scale']} · **C** = {CONFIG['params']['C']:g}
- **Noyau** : fidélité |⟨ψ(x)|ψ(x′)⟩|², simulation exacte
- **Classifieur** : SVM sur noyau précalculé
- **Entraînement** : 914 parcelles de Doukkala
""")
    m1, m2 = st.columns(2)
    m1.metric('Macro-F1 (validation)', '0,886')
    m2.metric('Exactitude', '88,7 %')
    st.markdown('### ⚠️ Domaine de validité')
    st.markdown("""<div class="tq-note">
Région de <b>Doukkala</b>, image du <b>25 avril 2024</b>, Sentinel-2 niveau L2A (réflectance ×10 000).
Hors de ce cadre (autre région, autre date, autre capteur), les prédictions ne sont pas garanties.
Trois cultures seulement : toute parcelle est classée dans l’une d’elles.</div>""", unsafe_allow_html=True)

tab_img, tab_csv, tab_demo, tab_help = st.tabs(['📷 Image de parcelle', '📄 Fichier de parcelles (CSV)',
                                                '🎯 Démo sur la validation', 'ℹ️ Aide'])

# ---------------------------------------------------------------- image
with tab_img:
    st.markdown('Déposez une **image multispectrale Sentinel-2 d’une seule parcelle** (GeoTIFF à 12 bandes '
                f'{", ".join(BANDS)}, ou 13 avec le NDVI).')
    up = st.file_uploader('Image de la parcelle', type=['tif', 'tiff', 'npy'], label_visibility='collapsed')
    samples = sorted((ROOT / 'app/samples').glob('*.tif'))
    if samples and not up:
        pick = st.selectbox('…ou essayez une image d’exemple', ['—'] + [p.name for p in samples])
        if pick != '—':
            p = ROOT / 'app/samples' / pick
            up = type('F', (), {'name': p.name, 'getvalue': lambda self, p=p: p.read_bytes()})()
    if up:
        try:
            img = read_image(up.getvalue(), up.name)
            field, n_valid, notes = summarize_field(img)
        except ValueError as e:
            st.error(str(e))
        else:
            for n in notes: st.info(n)

            def preview():
                st.markdown('##### Aperçu couleurs naturelles (B4-B3-B2)')
                st.image(rgb_preview(img), use_container_width=True, clamp=True)
                st.caption(f'{img.shape[0]}×{img.shape[1]} pixels · {img.shape[2]} bandes · {n_valid} pixels valides résumés par leur médiane')
            show_prediction(field, predictor, extra=preview)

# ---------------------------------------------------------------- CSV
with tab_csv:
    st.markdown('Un fichier CSV avec une ligne par parcelle et les colonnes '
                f'`{", ".join(FEATURES)}` (colonne `sample_id` facultative).')
    up_csv = st.file_uploader('Fichier CSV', type=['csv'], label_visibility='collapsed', key='csv')
    if up_csv:
        df = pd.read_csv(up_csv)
        missing = [c for c in FEATURES if c not in df.columns]
        if 'NDVI' in missing and all(b in df.columns for b in ('B4', 'B8')):
            df['NDVI'] = (df.B8 - df.B4) / (df.B8 + df.B4); missing.remove('NDVI')
        if missing:
            st.error(f'Colonnes manquantes : {", ".join(missing)}')
        elif not np.isfinite(df[FEATURES].to_numpy(float)).all():
            st.error('Le fichier contient des valeurs manquantes ou non numériques.')
        else:
            with st.spinner('Prédiction en cours…'):
                res = predictor.predict(df)
            out = pd.concat([df[['sample_id']] if 'sample_id' in df else pd.DataFrame(index=df.index),
                             res[['culture', 'score_alfalfa', 'score_maize', 'score_wheat', 'familier']]], axis=1)
            counts = res.culture.value_counts()
            cols = st.columns(4)
            cols[0].metric('Parcelles', len(res))
            for col, crop in zip(cols[1:], COLORS):
                col.metric(f'{ICONS[crop]} {crop}', int(counts.get(crop, 0)))
            if (~res.familier).any():
                st.warning(f'{int((~res.familier).sum())} parcelle(s) inhabituelle(s) par rapport aux données d’entraînement.')
            st.dataframe(out, hide_index=True, use_container_width=True, column_config={
                f'score_{c}': st.column_config.ProgressColumn(f'Score {LABELS_FR[c]}', min_value=0, max_value=1, format='percent')
                for c in ['alfalfa', 'maize', 'wheat']})
            st.download_button('⬇️ Télécharger les prédictions (CSV)', out.to_csv(index=False).encode('utf-8'),
                               'predictions_techquantum.csv', 'text/csv')

# ---------------------------------------------------------------- demo
with tab_demo:
    st.markdown('Choisissez une parcelle **de validation** (jamais vue à l’entraînement) et comparez la prédiction à la vérité terrain.')
    va = predictor.validation
    c1, c2 = st.columns([1, 2])
    crop_filter = c1.selectbox('Culture réelle', ['Toutes'] + list(COLORS))
    pool = va if crop_filter == 'Toutes' else va[va.target.map(LABELS_FR) == crop_filter]
    sid = c2.selectbox('Parcelle', pool.sample_id.tolist())
    field = va.loc[va.sample_id == sid, FEATURES].reset_index(drop=True)
    truth = LABELS_FR[va.loc[va.sample_id == sid, 'target'].iloc[0]]

    def verdict():
        row = predictor.predict(field).iloc[0]
        ok = row['culture'] == truth
        (st.success if ok else st.error)(f'Vérité terrain : **{ICONS[truth]} {truth}** — prédiction {"correcte ✅" if ok else "incorrecte ❌"}')
    show_prediction(field, predictor, extra=verdict)

# ---------------------------------------------------------------- help
with tab_help:
    st.markdown("""
#### Quelle image fournir ?
Une image **multispectrale Sentinel-2** recadrée sur **une seule parcelle**, au format GeoTIFF, avec les bandes dans l’ordre
**B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12** (le NDVI en 13ᵉ bande est accepté mais recalculé).
Valeurs de réflectance ×10 000 (niveau L2A) ou entre 0 et 1 (converties automatiquement).

Vous pouvez l’exporter depuis le **Copernicus Browser** ou **Google Earth Engine** (`COPERNICUS/S2_SR_HARMONIZED`).

#### Pourquoi pas une simple photo ?
Une photo couleur ne contient que 3 bandes (rouge, vert, bleu). Le modèle a besoin des bandes **infrarouges**
(B5 à B12), invisibles à l’œil, qui distinguent l’état de la végétation. Sans elles, aucune prédiction fiable n’est possible.

#### Comment l’image est-elle transformée ?
1. Les pixels invalides (nuls, négatifs, manquants) sont écartés.
2. Chaque bande est résumée par sa **médiane** sur la parcelle, puis le NDVI est calculé.
3. Les 13 valeurs passent par la PCA à 8 variables, puis par le circuit quantique à 8 qubits.
4. Le SVM compare l’état quantique obtenu à ceux des 914 parcelles d’entraînement.

#### Que signifie « parcelle inhabituelle » ?
La parcelle est moins semblable à toutes les parcelles d’entraînement que 98 % des parcelles de validation.
Le modèle risque alors de se tromper (autre région, nuages, sol nu, autre culture…).
""")
