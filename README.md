# Predicción de partidos de tenis

Modelo de aprendizaje automático entrenado con **832.242 partidos** para estimar
la probabilidad de victoria en un partido de tenis profesional.

No es una herramienta de apuestas: es un ejercicio de ciencia de datos sobre una
pregunta concreta — **¿cuánta información hace falta para predecir algo que
parece impredecible?**

---

## Resultados

|                        | ATP         | WTA         |
| ---------------------- | ----------- | ----------- |
| Precisión (test)       | **67,65 %** | **61,36 %** |
| AUC                    | 0,742       | 0,677       |
| Variables por partido  | 59          | 52          |
| Filas de entrenamiento | 269.109     | 71.757      |
| Jugadores con estado   | 25.810      | 20.806      |

**Cómo leer estos números.** El jugador mejor rankeado gana alrededor del 65 %
de los partidos, así que ese es el piso: acertar el 67,65 % significa que el
modelo aporta información por encima de lo evidente. El AUC mide si además
ordena bien los casos dudosos, que es lo que de verdad importa.

**La diferencia entre ATP y WTA no es un defecto del modelo.** El circuito
femenino es genuinamente menos predecible: menos partidos históricos por
jugadora y más variabilidad entre torneos. El modelo lo refleja en vez de
esconderlo.

---

## Cómo se evalúa sin hacer trampa

Entrenar con todos los datos y medir sobre esos mismos datos infla el resultado
sin que uno se dé cuenta: el modelo memoriza en vez de predecir.

Acá la separación es **temporal**: se entrena solo con partidos hasta 2021 y se
evalúa con los posteriores. El modelo nunca vio los partidos con los que se le
mide. Es la única forma honesta de estimar cómo se comportaría con un partido
que todavía no se jugó.

Además se aplican **pesos temporales** con vida media de 3 años: un partido de
2012 pesa menos que uno de 2024, porque el tenis cambia.

---

## Las variables

Las 59 variables por partido salen de combinar cinco fuentes distintas:

- **Elo propio**, calculado desde cero y separado por superficie
- **Forma reciente**: resultados, rachas y descanso entre partidos
- **Estadísticas de saque y resto** de los últimos encuentros
- **Historial directo** entre los dos jugadores
- **Contexto**: superficie, categoría del torneo, ronda

---

## Lo más difícil no fue el modelo

Fue **juntar los datos**. No existe una base única de tenis profesional, así que
hay cinco fuentes que no se hablan entre sí:

| Fuente                    | Aporta                                  |
| ------------------------- | --------------------------------------- |
| Sportradar                | 24.449 partidos recientes con saque     |
| API-Tennis                | partidos del día y rankings             |
| tennis-data.co.uk         | histórico desde el año 2000             |
| Jeff Sackmann (charting)  | estadísticas punto por punto            |
| Rankings ATP/WTA          | posiciones semanales                    |

Cada una nombra a los jugadores distinto, cubre años distintos y usa códigos de
torneo distintos. Reconciliar eso —decidir que "R. Nadal", "Nadal R." y "Rafael
Nadal" son la misma persona, en bases con decenas de miles de nombres— fue la
mayor parte del trabajo.

---

## Estructura

```
src/
  ingestion/   descarga y normalización de las cinco fuentes
  features/    Elo, forma, estadísticas de saque, punto por punto
  models/      entrenamiento y predicción (ATP y WTA por separado)
  dashboard/   interfaz en Streamlit
  agents/      actualización diaria automática
```

---

## Correr el proyecto

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

**Los datos y los modelos entrenados no están en este repositorio**: son 525 MB,
y provienen de proveedores cuyas licencias no permiten redistribuirlos. El
código que los descarga y procesa sí está completo en `src/ingestion`, así que
el pipeline es reproducible desde cero.

---

## Stack

Python · XGBoost · scikit-learn · Optuna (optimización de hiperparámetros) ·
pandas · SQLite · Streamlit

---

*Felipe Massuh · 2026*
