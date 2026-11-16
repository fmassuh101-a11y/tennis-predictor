"""
Panel del modelo de predicción de tenis.

QUÉ MUESTRA Y QUÉ NO
Este panel es para explicar el modelo: con qué datos se entrenó, qué tan bien
acierta, y qué predice para un partido concreto.

NO muestra cuotas, casas de apuestas ni recomendaciones de apostar. La versión
anterior tenía una sección de "picks del día" con cuotas y comparaciones
contra casas de apuestas; se quitó entera. El trabajo detrás es análisis
predictivo —probabilidades calibradas sobre 832.242 partidos— y presentarlo
como un sistema de apuestas lo describía mal.

También se quitó esa sección porque mostraba datos congelados de marzo de
2026 sin avisar, lo que daba una impresión falsa de estar al día.
"""

import os
import sys
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.predict import predict_match, load_state, find_player  # noqa: E402

try:
    from src.models.predict_wta import predict_wta_match, load_wta_state  # noqa: E402
    HAY_WTA = True
except Exception:
    HAY_WTA = False


st.set_page_config(page_title="Predicción de partidos de tenis",
                   page_icon="🎾", layout="wide")

# ─────────────────────────────────────────────────────────────────────────
# Cifras del modelo. Salen de la evaluación real de los .pkl entrenados, no
# están puestas a mano: si se reentrena, hay que actualizarlas acá.
# ─────────────────────────────────────────────────────────────────────────
# El MODELO se entrenó con 832.242 partidos históricos; esa cifra no cambia.
# El ESTADO DE LOS JUGADORES —Elo, forma, descanso— se reconstruye aparte y
# es lo que hay que mantener al día: con un estado viejo, el modelo predice
# bien pero sobre jugadores que ya no están en ese nivel.
ATP = {"precision": 67.65, "auc": 0.742, "partidos": 832242,
       "entrenamiento": 269109, "variables": 59, "jugadores": 988,
       "periodo": "1991 – 2026"}
WTA = {"precision": 61.36, "auc": 0.677, "partidos": 530000,
       "entrenamiento": 71757, "variables": 52, "jugadores": 20806,
       "periodo": "2003 – 2026"}

st.markdown("""
<style>
  .big {font-size:2.6rem;font-weight:800;line-height:1.1;margin:0}
  .sub {color:#8a8a8e;font-size:.95rem;margin-top:.2rem}
  .caja {background:#f6f7f9;border-radius:12px;padding:1.1rem 1.3rem;margin:.4rem 0}
  .ganador {background:linear-gradient(135deg,#1d6fd4,#1856a8);color:#fff;
            border-radius:14px;padding:1.5rem 1.8rem;margin:1rem 0}
</style>""", unsafe_allow_html=True)


# ═════════════════════════ BARRA LATERAL ═════════════════════════
with st.sidebar:
    st.markdown("## 🎾 Predicción de tenis")
    st.caption("Modelo de aprendizaje automático sobre partidos profesionales")
    pagina = st.radio("Sección", [
        "Rendimiento del modelo",
        "Predecir un partido (ATP)",
        "Predecir un partido (WTA)",
        "Perfil de jugador",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Cómo se evalúa**")
    st.caption(
        "Se entrena solo con partidos hasta 2021 y se mide con los "
        "posteriores. El modelo nunca ve los partidos con los que se le "
        "evalúa — es la única forma honesta de saber si predice o memoriza."
    )


# ═════════════════════════ RENDIMIENTO ═════════════════════════
if pagina == "Rendimiento del modelo":
    st.markdown('<p class="big">Rendimiento del modelo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Medido sobre partidos que el modelo nunca vio '
                'durante el entrenamiento.</p>', unsafe_allow_html=True)
    st.write("")

    for nombre, d in [("Circuito ATP", ATP), ("Circuito WTA", WTA)]:
        st.markdown(f"### {nombre}")
        c = st.columns(4)
        c[0].metric("Precisión", f"{d['precision']} %")
        c[1].metric("AUC", f"{d['auc']}")
        c[2].metric("Partidos procesados", f"{d['partidos']:,}".replace(",", "."))
        c[3].metric("Variables por partido", d["variables"])
        c2 = st.columns(3)
        c2[0].metric("Filas de entrenamiento", f"{d['entrenamiento']:,}".replace(",", "."))
        c2[1].metric("Jugadores con estado", f"{d['jugadores']:,}".replace(",", "."))
        c2[2].metric("Periodo cubierto", d["periodo"])
        st.write("")

    st.info(
        "**Estado de los jugadores actualizado al 14 de noviembre de 2026.** "
        "El modelo está entrenado con 832.242 partidos históricos; el Elo, la "
        "forma y el descanso de cada jugador se recalculan aparte con los "
        "resultados del circuito principal, para que la predicción use el "
        "nivel actual y no el de hace meses."
    )

    st.markdown("---")
    st.markdown("### Cómo leer estos números")
    st.markdown("""
El jugador mejor rankeado gana alrededor del **65 %** de los partidos. Ese es
el piso: acertar el 67,65 % significa que el modelo aporta información por
encima de lo evidente. El **AUC** mide algo distinto y más exigente — si el
modelo además ordena bien los casos dudosos, que es donde se juega el valor
real de una predicción.

La diferencia entre ATP y WTA **no es un defecto del modelo**. El circuito
femenino es genuinamente menos predecible: hay menos partidos históricos por
jugadora y más variabilidad entre torneos. El modelo lo refleja en vez de
esconderlo.
""")

    st.markdown("### De dónde salen los datos")
    st.table(pd.DataFrame({
        "Fuente": ["Sportradar", "API-Tennis", "tennis-data.co.uk",
                   "Jeff Sackmann (charting)", "Rankings ATP / WTA"],
        "Qué aporta": [
            "24.449 partidos recientes con estadísticas de saque",
            "Partidos del día y rankings actualizados",
            "Histórico desde el año 2000",
            "Estadísticas punto por punto",
            "Posiciones semanales",
        ],
    }))
    st.caption(
        "Lo más difícil del proyecto no fue el modelo: fue que estas cinco "
        "fuentes no se hablan entre sí. Cada una nombra a los jugadores "
        "distinto, cubre años distintos y usa códigos de torneo distintos. "
        "Reconciliar eso, en bases con decenas de miles de nombres, fue la "
        "mayor parte del trabajo."
    )

    st.markdown("### Las variables")
    st.markdown("""
- **Elo propio**, calculado desde cero y separado por superficie
- **Forma reciente**: resultados, rachas y días de descanso
- **Estadísticas de saque y resto** de los últimos encuentros
- **Historial directo** entre los dos jugadores
- **Contexto**: superficie, categoría del torneo y ronda

Se aplican además **pesos temporales** con vida media de tres años: un partido
de 2012 pesa menos que uno de 2024, porque el tenis cambia.
""")


# ═════════════════════════ PREDECIR ═════════════════════════
elif pagina.startswith("Predecir"):
    es_wta = "WTA" in pagina
    if es_wta and not HAY_WTA:
        st.error("El modelo WTA no está disponible en este entorno.")
        st.stop()

    d = WTA if es_wta else ATP
    st.markdown(f'<p class="big">Predecir un partido · {"WTA" if es_wta else "ATP"}</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub">El modelo devuelve una probabilidad calibrada, '
                'no solo un ganador.</p>', unsafe_allow_html=True)
    st.write("")

    @st.cache_data(show_spinner=False)
    def lista_jugadores(wta: bool):
        estado = load_wta_state() if wta else load_state()
        try:
            return sorted(estado.keys())
        except Exception:
            return []

    jugadores = lista_jugadores(es_wta)

    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("Jugador 1", jugadores, index=0 if jugadores else None,
                          key=f"p1{es_wta}")
    with c2:
        p2 = st.selectbox("Jugador 2", jugadores, index=1 if len(jugadores) > 1 else None,
                          key=f"p2{es_wta}")

    c = st.columns(4)
    superficie = c[0].selectbox("Superficie", ["Hard", "Clay", "Grass", "Carpet"])
    nivel = c[1].selectbox("Categoría", ["Grand Slam", "Masters 1000",
                                         "ATP250/500", "Challenger"])
    al_mejor_de = c[2].selectbox("Al mejor de", [3, 5])
    ronda = c[3].selectbox("Ronda", ["R128", "R64", "R32", "R16", "QF", "SF", "F"])

    if st.button("Calcular probabilidad", type="primary", use_container_width=True):
        if p1 == p2:
            st.error("Elige dos jugadores distintos.")
        else:
            with st.spinner("Calculando…"):
                fn = predict_wta_match if es_wta else predict_match
                try:
                    r = fn(p1, p2, superficie, level=nivel, best_of=al_mejor_de,
                           round_name=ronda, verbose=False)
                except TypeError:
                    r = fn(p1, p2, superficie, verbose=False)

            if isinstance(r, dict) and "error" in r:
                st.error(r["error"])
            else:
                ganador = r.get("predicted_winner", "—")
                prob = float(r.get("confidence", 0)) * 100

                st.markdown(f"""
<div class="ganador">
  <div style="opacity:.85;font-size:.9rem;text-transform:uppercase;
              letter-spacing:.08em">Predicción</div>
  <div style="font-size:2rem;font-weight:800;margin:.3rem 0">{ganador}</div>
  <div style="font-size:1.1rem">Probabilidad de victoria: <b>{prob:.1f} %</b></div>
</div>""", unsafe_allow_html=True)

                fig = go.Figure(go.Bar(
                    x=[prob, 100 - prob],
                    y=[r.get("p1_name", p1), r.get("p2_name", p2)],
                    orientation="h",
                    marker_color=["#1d6fd4", "#c9ced6"],
                    text=[f"{prob:.1f} %", f"{100-prob:.1f} %"],
                    textposition="auto",
                ))
                fig.update_layout(height=180, showlegend=False,
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis=dict(range=[0, 100], title="Probabilidad"))
                st.plotly_chart(fig, use_container_width=True)

                # Los días sin jugar avisan cuándo la predicción es menos
                # confiable: si el último partido en la base es de hace más
                # de medio año, el estado del jugador está desactualizado.
                for lado, nom in [("p1", r.get("p1_name", p1)), ("p2", r.get("p2_name", p2))]:
                    dias = r.get(f"{lado}_rest_days") or 0
                    if dias > 200:
                        st.warning(
                            f"El último partido de **{nom}** en la base es de "
                            f"hace {dias} días. La predicción se apoya solo en "
                            f"su historial, no en su forma actual."
                        )

                with st.expander("Ver el detalle del cálculo"):
                    st.json({k: v for k, v in r.items()
                             if not k.startswith("_") and "odd" not in k.lower()
                             and "bet" not in k.lower() and "edge" not in k.lower()})


# ═════════════════════════ PERFIL ═════════════════════════
else:
    st.markdown('<p class="big">Perfil de jugador</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Estado calculado a partir de su historial.</p>',
                unsafe_allow_html=True)
    st.write("")

    @st.cache_data(show_spinner=False)
    def estado_atp():
        return load_state()

    estado = estado_atp()
    nombres = sorted(estado.keys()) if hasattr(estado, "keys") else []
    quien = st.selectbox("Jugador", nombres)

    if quien:
        p = estado.get(quien, {})
        if isinstance(p, dict):
            c = st.columns(4)
            c[0].metric("Elo general", f"{p.get('elo', 0):.0f}" if p.get("elo") else "—")
            c[1].metric("Elo en dura", f"{p.get('elo_hard', 0):.0f}" if p.get("elo_hard") else "—")
            c[2].metric("Elo en tierra", f"{p.get('elo_clay', 0):.0f}" if p.get("elo_clay") else "—")
            c[3].metric("Partidos", p.get("matches", "—"))
            with st.expander("Ver todo el estado"):
                st.json(p)
        else:
            st.info("Sin datos para ese jugador.")


st.divider()
st.caption(
    f"Felipe Massuh · {date.today().year} · "
    "Análisis predictivo deportivo con aprendizaje automático"
)
