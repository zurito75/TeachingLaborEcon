"""
impuestos_nomina_model.py
==========================
Modelo de incidencia de los impuestos a la nómina sobre el empleo y los
salarios, en un mercado laboral competitivo.

Curso: EC0734 Economía Laboral (EAFIT) — Semana 7, Instituciones y
políticas del mercado laboral.
Autor: Leo Morales (con apoyo de Claude/AIRA)

Marco teórico combinado a partir de tres referencias (ver la nota
comparativa en Notas/nota_comparativa_impuestos_nomina.md):

  - Núcleo del modelo (mercado competitivo, notación t/b/eta/phi):
    Kugler y Kugler (2009), "Labor Market Effects of Payroll Taxes in
    Developing Countries: Evidence from Colombia", ecs. (1)-(2), sección
    "II. Theoretical Effects of Payroll Taxes".
  - Extensión de salario mínimo vinculante:
    Kugler y Kugler (2009), sección II; Becerra y Morales (2025), ec. (7).
  - Extensión de sector informal:
    Becerra y Morales (2025), "Labor Demand Responses to Payroll Taxes in
    an Economy with Wage Rigidity: Evidence from Colombia" (BanRep WP
    1297), Apéndice A "Conceptual framework", ec. (8).

Este script:
  1) Implementa en forma cerrada la demanda y la oferta de trabajo con
     elasticidad constante, D(w(1+t)) = S(w(1+bt)), y resuelve el
     equilibrio EXACTO (no solo la aproximación marginal) para cualquier
     tasa de impuesto t, en los tres regímenes:
       (a) mercado competitivo (traslado parcial a salarios/empleo)
       (b) salario mínimo vinculante
       (c) mercado con sector informal
  2) Deriva también las fórmulas de estática comparativa MARGINAL
     (elasticidad puntual, dw/w /dt y dL/L /dt) para cada régimen, y
     verifica numéricamente que coinciden con la derivada exacta
     (diferencias finitas) del equilibrio calculado en (1).
  3) Calibra el modelo con la reforma tributaria de 2013 en Colombia
     (Ley 1607 de 2012): recorte de 13.5 puntos porcentuales en los
     aportes patronales a la nómina (salud 8.5pp + SENA 2pp + ICBF 3pp),
     partiendo de una carga parafiscal legal total de 52.3% del salario
     antes de la reforma (Hernández 2012, tabla reproducida en Morales y
     Medina 2017, tabla 1), y con la elasticidad de demanda de trabajo
     formal estimada por Becerra y Morales (2025): entre -0.53 y -0.87.
  4) Genera las figuras de estática comparativa (diagrama de brecha
     tributaria/"tax wedge" para los tres regímenes) en español, para
     usarlas en el deck de Beamer.

NOTA: Todo lo que sigue es estática comparativa entre equilibrios (no
dinámica de ajuste). No se usa scipy (solo numpy/matplotlib) para que el
script corra igual en la máquina del curso.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (6.4, 4.8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


# ---------------------------------------------------------------------------
# 0. Bisección genérica (evitamos scipy: solo numpy/matplotlib)
# ---------------------------------------------------------------------------

def _biseccion(f, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Encuentra una raíz de f en [lo, hi], asumiendo f(lo) y f(hi) de signo
    opuesto. f debe ser monótona en el intervalo (nuestras funciones lo son)."""
    flo = f(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if (fmid > 0) == (flo > 0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 1. Parámetros del modelo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Parametros:
    """Parámetros estructurales y de calibración."""
    # --- impuesto a la nómina (carga legal del empleador, % del salario) ---
    t0: float = 0.523   # antes de la reforma de 2013 (Hernández 2012)
    t1: float = 0.388   # después de la reforma de 2013 (t0 - 13.5pp)

    # --- núcleo competitivo ---
    b: float = 0.50     # valoración del trabajador de los beneficios (b en [0,1])
    phi: float = 0.40   # elasticidad de oferta de trabajo (ilustrativa, φ>0)
    eta: float = -0.70  # elasticidad de demanda de trabajo (Becerra y Morales 2025: [-0.87,-0.53])

    # --- salario mínimo vinculante ---
    w_min_rel: float = 1.15  # salario mínimo relativo al salario competitivo w0* (>1 => vinculante)

    # --- sector informal ---
    s_formal0: float = 0.50  # participación del empleo formal en el empleo total, L_m, antes de la reforma
    eta_informal: float = -0.30  # elasticidad de demanda de trabajo informal (no paga el impuesto)

    # --- normalización (no son "datos", solo escala) ---
    w0: float = 1.0   # salario competitivo normalizado a 1 en t0
    L0: float = 1.0   # empleo normalizado a 1 en t0 (o L_m en el régimen informal)


# ---------------------------------------------------------------------------
# 2. Régimen (a): mercado competitivo — equilibrio EXACTO
# ---------------------------------------------------------------------------
#
# Demanda:  D(x) = K_D * x^eta,           x = w(1+t)      (costo total p/ la firma)
# Oferta:   S(x) = K_S * x^phi,           x = w(1+b t)     (compensación percibida)
# K_D, K_S se calibran para que, en t=t0, w=w0 y L=L0 (D=S=L0).

def _constantes_demanda_oferta(p: Parametros) -> tuple[float, float]:
    K_D = p.L0 / (p.w0 * (1 + p.t0)) ** p.eta
    K_S = p.L0 / (p.w0 * (1 + p.b * p.t0)) ** p.phi
    return K_D, K_S


def equilibrio_competitivo(t: float, p: Parametros) -> dict:
    """Salario y empleo de equilibrio EXACTOS (no solo la aproximación
    marginal) para una tasa de impuesto t, resolviendo D(w(1+t))=S(w(1+bt))."""
    K_D, K_S = _constantes_demanda_oferta(p)
    # w^(eta-phi) = K_S (1+bt)^phi / (K_D (1+t)^eta)  =>  log w = (1/(eta-phi)) * [...]
    log_rhs = (np.log(K_S) + p.phi * np.log(1 + p.b * t)) - (np.log(K_D) + p.eta * np.log(1 + t))
    w = np.exp(log_rhs / (p.eta - p.phi))
    L = K_D * (w * (1 + t)) ** p.eta
    return {"w": w, "L": L, "regimen": "competitivo"}


def formulas_marginales_competitivo(t: float, p: Parametros) -> dict:
    """Elasticidad puntual (derivada) evaluada en t: dw/w /dt y dL/L /dt.

    Derivación (paso a paso en las slides): se diferencia totalmente
    D(w(1+t))=S(w(1+bt)) en logaritmos, usando que d ln D = eta * d ln[w(1+t)]
    y d ln S = phi * d ln[w(1+bt)]:

        (dw/w)/dt = [b(1+t)phi - (1+bt)eta] / [(eta-phi)(1+bt)(1+t)]
        (dL/L)/dt = eta*phi*(b-1) / [(eta-phi)(1+bt)(1+t)]

    (Coincide exactamente con la ec. (5)-(6) de Becerra y Morales 2025,
    con alpha=b.)
    """
    eta, phi, b = p.eta, p.phi, p.b
    dw_dt = (b * (1 + t) * phi - (1 + b * t) * eta) / ((eta - phi) * (1 + b * t) * (1 + t))
    dL_dt = eta * phi * (b - 1) / ((eta - phi) * (1 + b * t) * (1 + t))
    return {"dw_w_dt": dw_dt, "dL_L_dt": dL_dt}


# ---------------------------------------------------------------------------
# 3. Régimen (b): salario mínimo vinculante
# ---------------------------------------------------------------------------
#
# El salario queda fijo en w_bar (no puede ajustar a la baja). El empleo se
# lee directamente de la curva de demanda: L = D(w_bar (1+t)) = K_D [w_bar(1+t)]^eta.
# Es vinculante mientras w_bar > w*(t) (el salario competitivo NO restringido).

def equilibrio_salario_minimo(t: float, p: Parametros) -> dict:
    K_D, _ = _constantes_demanda_oferta(p)
    w_bar = p.w_min_rel * p.w0  # nivel del salario mínimo (fijo, no depende de t)
    L = K_D * (w_bar * (1 + t)) ** p.eta
    eq_no_restringido = equilibrio_competitivo(t, p)
    vinculante = w_bar > eq_no_restringido["w"]
    return {"w": w_bar, "L": L, "regimen": "salario_minimo",
            "vinculante": bool(vinculante), "w_no_restringido": eq_no_restringido["w"]}


def formula_marginal_salario_minimo(t: float, p: Parametros) -> dict:
    """dw/w /dt = 0 (el salario no se mueve); dL/L /dt = eta/(1+t)
    (Becerra y Morales 2025, ec. 7 — coincide con el caso límite de Kugler
    y Kugler 2009 cuando el salario mínimo es vinculante)."""
    return {"dw_w_dt": 0.0, "dL_L_dt": p.eta / (1 + t)}


# ---------------------------------------------------------------------------
# 4. Régimen (c): sector informal
# ---------------------------------------------------------------------------
#
# Oferta agregada FIJA (perfectamente inelástica) L_m = L0. Dos demandas:
#   formal:   Df(x) = K_Df * x^eta_f,     x = w(1+t)
#   informal: Di(w) = K_Di * w^eta_i      (no paga el impuesto)
# Equilibrio: Df(w(1+t)) + Di(w) = L_m.  Se resuelve w por bisección
# (no hay, en general, forma cerrada si eta_f != eta_i).

def _constantes_informal(p: Parametros) -> tuple[float, float]:
    L_m = p.L0
    Df0 = p.s_formal0 * L_m
    Di0 = (1 - p.s_formal0) * L_m
    K_Df = Df0 / (p.w0 * (1 + p.t0)) ** p.eta
    K_Di = Di0 / (p.w0) ** p.eta_informal
    return K_Df, K_Di


def equilibrio_informal(t: float, p: Parametros) -> dict:
    K_Df, K_Di = _constantes_informal(p)
    L_m = p.L0

    def exceso(w: float) -> float:
        Df = K_Df * (w * (1 + t)) ** p.eta
        Di = K_Di * (w) ** p.eta_informal
        return Df + Di - L_m

    w = _biseccion(exceso, 1e-6, 10 * p.w0)
    Df = K_Df * (w * (1 + t)) ** p.eta
    Di = K_Di * (w) ** p.eta_informal
    return {"w": w, "Lf": Df, "Li": Di, "L_m": L_m, "regimen": "informal"}


def formula_marginal_informal(t: float, p: Parametros) -> dict:
    """(dLf/Lf)/dt = eta_f * eta_i * s_i / [(1+t) (eta_f s_f + eta_i s_i)],
    con s_f, s_i las participaciones formal/informal EN t (no solo en t0).
    Derivación: se diferencia Df(w(1+t))+Di(w)=L_m (nivel fijo) usando las
    elasticidades eta_f=eta, eta_i=eta_informal (ver slides para el paso a
    paso). Es la versión de dos sectores de la ec. (8) de Becerra y Morales
    (2025)."""
    eq = equilibrio_informal(t, p)
    s_f = eq["Lf"] / eq["L_m"]
    s_i = eq["Li"] / eq["L_m"]
    eta_f, eta_i = p.eta, p.eta_informal
    dw_dt = -eta_f * s_f / ((eta_f * s_f + eta_i * s_i) * (1 + t))
    dLf_dt = eta_f * eta_i * s_i / ((1 + t) * (eta_f * s_f + eta_i * s_i))
    return {"dw_w_dt": dw_dt, "dLf_Lf_dt": dLf_dt, "s_f": s_f, "s_i": s_i}


# ---------------------------------------------------------------------------
# 5. Verificación numérica (derivada exacta por diferencias finitas
#    contra las fórmulas cerradas — y casos polares conocidos)
# ---------------------------------------------------------------------------

def _dlx_dt_diferencias_finitas(f_w_L, t: float, p: Parametros, h: float = 1e-6):
    eq_mas = f_w_L(t + h, p)
    eq_menos = f_w_L(t - h, p)
    dw_dt = (np.log(eq_mas["w"]) - np.log(eq_menos["w"])) / (2 * h)
    return dw_dt


def verificar_formulas() -> None:
    p = Parametros()
    print("=" * 72)
    print("VERIFICACIÓN 1 — Régimen competitivo: derivada exacta (dif. finitas)")
    print("                  vs. fórmula cerrada, en t=t0")
    print("=" * 72)
    dw_dt_exacto = _dlx_dt_diferencias_finitas(equilibrio_competitivo, p.t0, p)
    eq0 = equilibrio_competitivo(p.t0, p)
    eq_mas = equilibrio_competitivo(p.t0 + 1e-6, p)
    eq_menos = equilibrio_competitivo(p.t0 - 1e-6, p)
    dL_dt_exacto = (np.log(eq_mas["L"]) - np.log(eq_menos["L"])) / (2e-6)
    formula = formulas_marginales_competitivo(p.t0, p)
    print(f"  dw/w /dt   exacto={dw_dt_exacto:+.6f}   fórmula={formula['dw_w_dt']:+.6f}")
    print(f"  dL/L /dt   exacto={dL_dt_exacto:+.6f}   fórmula={formula['dL_L_dt']:+.6f}")
    assert abs(dw_dt_exacto - formula["dw_w_dt"]) < 1e-4
    assert abs(dL_dt_exacto - formula["dL_L_dt"]) < 1e-4
    print("  OK: la fórmula cerrada coincide con la derivada numérica.\n")

    print("=" * 72)
    print("VERIFICACIÓN 2 — Casos polares del régimen competitivo")
    print("=" * 72)
    for nombre, p_caso in [
        ("b=1 (beneficio valorado 1 a 1)", replace(p, b=1.0)),
        ("phi=0 (oferta perfectamente inelástica)", replace(p, phi=1e-8)),
    ]:
        f = formulas_marginales_competitivo(p.t0, p_caso)
        print(f"  {nombre}: dL/L /dt = {f['dL_L_dt']:+.8f}  (debe ser ~0)")
        assert abs(f["dL_L_dt"]) < 1e-3
    # Caso "eta -> -infinito" (demanda muy elástica) con b<1: el efecto
    # empleo NO se anula en general (a diferencia de lo que sugiere una
    # lectura rápida de Kugler y Kugler 2009) — solo se anula si además
    # b=1 o phi=0. Ver nota en las slides.
    f_eta_grande = formulas_marginales_competitivo(p.t0, replace(p, eta=-1e6))
    limite_teorico = p.phi * (p.b - 1) / ((1 + p.b * p.t0) * (1 + p.t0))
    print(f"  eta->-inf, b={p.b} (no polar): dL/L /dt = {f_eta_grande['dL_L_dt']:+.6f}"
          f"   límite teórico phi(b-1)/[(1+bt)(1+t)] = {limite_teorico:+.6f}")
    assert abs(f_eta_grande["dL_L_dt"] - limite_teorico) < 1e-3
    print("  OK: confirma que 'demanda perfectamente elástica' por sí sola NO")
    print("      basta para anular el efecto empleo si b<1 y phi>0.\n")

    print("=" * 72)
    print("VERIFICACIÓN 3 — Régimen de salario mínimo vinculante")
    print("=" * 72)
    dw_dt_exacto_sm = _dlx_dt_diferencias_finitas(
        lambda t, p: equilibrio_salario_minimo(t, p), p.t0, p)
    eq_mas = equilibrio_salario_minimo(p.t0 + 1e-6, p)
    eq_menos = equilibrio_salario_minimo(p.t0 - 1e-6, p)
    dL_dt_exacto_sm = (np.log(eq_mas["L"]) - np.log(eq_menos["L"])) / (2e-6)
    formula_sm = formula_marginal_salario_minimo(p.t0, p)
    print(f"  dw/w /dt   exacto={dw_dt_exacto_sm:+.6f}   fórmula={formula_sm['dw_w_dt']:+.6f}")
    print(f"  dL/L /dt   exacto={dL_dt_exacto_sm:+.6f}   fórmula={formula_sm['dL_L_dt']:+.6f}")
    assert abs(dL_dt_exacto_sm - formula_sm["dL_L_dt"]) < 1e-4
    print("  OK: coincide con eta/(1+t) (ec. 7 de Becerra y Morales 2025).\n")

    print("=" * 72)
    print("VERIFICACIÓN 4 — Régimen de sector informal")
    print("=" * 72)
    eq_mas = equilibrio_informal(p.t0 + 1e-6, p)
    eq_menos = equilibrio_informal(p.t0 - 1e-6, p)
    dLf_dt_exacto = (np.log(eq_mas["Lf"]) - np.log(eq_menos["Lf"])) / (2e-6)
    formula_inf = formula_marginal_informal(p.t0, p)
    print(f"  dLf/Lf /dt   exacto={dLf_dt_exacto:+.6f}   fórmula={formula_inf['dLf_Lf_dt']:+.6f}")
    assert abs(dLf_dt_exacto - formula_inf["dLf_Lf_dt"]) < 1e-4
    # caso polar: sin sector informal (s_i -> 0), el empleo formal no cambia
    p_sin_informal = replace(p, s_formal0=1 - 1e-6)
    f_sin_informal = formula_marginal_informal(p.t0, p_sin_informal)
    print(f"  s_informal->0: dLf/Lf /dt = {f_sin_informal['dLf_Lf_dt']:+.8f}  (debe ser ~0)")
    assert abs(f_sin_informal["dLf_Lf_dt"]) < 1e-3
    print("  OK: coincide con la derivada numérica y con el caso polar esperado.\n")


# ---------------------------------------------------------------------------
# 6. Calibración con la reforma de 2013 (Ley 1607 de 2012)
# ---------------------------------------------------------------------------

def resumen_calibracion_2013() -> None:
    p = Parametros()
    print("=" * 72)
    print("CALIBRACIÓN — reforma tributaria de 2013 en Colombia (Ley 1607/2012)")
    print("=" * 72)
    print(f"  t0 (antes, 2012): {p.t0:.3f}   t1 (después, 2014): {p.t1:.3f}"
          f"   Δt = {p.t1 - p.t0:+.3f}  (-13.5 pp: salud 8.5 + SENA 2 + ICBF 3)")
    print(f"  Elasticidad de demanda formal (Becerra y Morales 2025): eta = {p.eta:.2f}"
          f"  (rango estimado: [-0.87, -0.53])")
    print()

    print("  (a) Régimen competitivo — cambio EXACTO t0 -> t1:")
    eq0 = equilibrio_competitivo(p.t0, p)
    eq1 = equilibrio_competitivo(p.t1, p)
    dw_pct = 100 * (eq1["w"] / eq0["w"] - 1)
    dL_pct = 100 * (eq1["L"] / eq0["L"] - 1)
    dt = p.t1 - p.t0
    approx = formulas_marginales_competitivo(p.t0, p)
    print(f"      salario:  {dw_pct:+.2f}%   (aprox. marginal: {100*approx['dw_w_dt']*dt:+.2f}%)")
    print(f"      empleo:   {dL_pct:+.2f}%   (aprox. marginal: {100*approx['dL_L_dt']*dt:+.2f}%)")

    print("  (b) Régimen de salario mínimo vinculante:")
    eqm0 = equilibrio_salario_minimo(p.t0, p)
    eqm1 = equilibrio_salario_minimo(p.t1, p)
    print(f"      vinculante en t0: {eqm0['vinculante']}   en t1: {eqm1['vinculante']}")
    dLm_pct = 100 * (eqm1["L"] / eqm0["L"] - 1)
    print(f"      salario: 0.00% (fijo)   empleo: {dLm_pct:+.2f}%")

    print("  (c) Régimen de sector informal:")
    eqi0 = equilibrio_informal(p.t0, p)
    eqi1 = equilibrio_informal(p.t1, p)
    dwi_pct = 100 * (eqi1["w"] / eqi0["w"] - 1)
    dLfi_pct = 100 * (eqi1["Lf"] / eqi0["Lf"] - 1)
    print(f"      salario (ambos sectores): {dwi_pct:+.2f}%   empleo formal: {dLfi_pct:+.2f}%"
          f"   (empleo formal/total pasa de {eqi0['Lf']/eqi0['L_m']:.1%} a {eqi1['Lf']/eqi1['L_m']:.1%})")
    print()


# ---------------------------------------------------------------------------
# 7. Figuras: diagrama de brecha tributaria ("tax wedge")
# ---------------------------------------------------------------------------
#
# En el espacio (L, w): la curva de demanda "inversa" (precio que el
# empleador está dispuesto a pagar, neto del impuesto) es
#     w_D(L) = [L/K_D]^(1/eta) / (1+t)
# y la curva de oferta "inversa" (compensación que exige el trabajador,
# en términos del salario nominal) es
#     w_S(L) = [L/K_S]^(1/phi) / (1+b t)

def _w_demanda(L, t, p, K_D=None):
    if K_D is None:
        K_D, _ = _constantes_demanda_oferta(p)
    return (L / K_D) ** (1 / p.eta) / (1 + t)


def _w_oferta(L, t, p, K_S=None):
    if K_S is None:
        _, K_S = _constantes_demanda_oferta(p)
    return (L / K_S) ** (1 / p.phi) / (1 + p.b * t)


def figura_competitivo(p: Parametros, archivo: str) -> None:
    K_D, K_S = _constantes_demanda_oferta(p)
    eq0 = equilibrio_competitivo(p.t0, p)
    eq1 = equilibrio_competitivo(p.t1, p)

    L_lo = min(eq0["L"], eq1["L"]) * 0.6
    L_hi = max(eq0["L"], eq1["L"]) * 1.4
    L_grid = np.linspace(L_lo, L_hi, 400)
    w_hi = max(eq0["w"], eq1["w"]) * 1.8

    fig, ax = plt.subplots()
    ax.plot(L_grid, _w_demanda(L_grid, p.t0, p, K_D), color="#1f4e79", lw=2,
            label=r"Demanda, $w(1+t_0)$ — antes")
    ax.plot(L_grid, _w_oferta(L_grid, p.t0, p, K_S), color="#c0392b", lw=2,
            label=r"Oferta, $w(1+bt_0)$ — antes")
    ax.plot(L_grid, _w_demanda(L_grid, p.t1, p, K_D), color="#1f4e79", lw=2, ls="--",
            label=r"Demanda, $w(1+t_1)$ — después (2013)")
    ax.plot(L_grid, _w_oferta(L_grid, p.t1, p, K_S), color="#c0392b", lw=2, ls="--",
            label=r"Oferta, $w(1+bt_1)$ — después (2013)")

    ax.plot([eq0["L"]], [eq0["w"]], "o", color="black", ms=7, zorder=5)
    ax.plot([eq1["L"]], [eq1["w"]], "s", color="black", ms=7, zorder=5)
    ax.annotate("equilibrio\nantes", (eq0["L"], eq0["w"]), textcoords="offset points",
                xytext=(-55, -8), fontsize=9)
    ax.annotate("equilibrio\ndespués", (eq1["L"], eq1["w"]), textcoords="offset points",
                xytext=(8, -4), fontsize=9)

    ax.set_xlim(L_lo, L_hi)
    ax.set_ylim(0, w_hi)
    ax.set_xlabel("Empleo, $L$")
    ax.set_ylabel("Salario, $w$")
    ax.set_title("Régimen competitivo: recorte del impuesto a la nómina (2013)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(archivo, dpi=200)
    plt.close(fig)
    print(f"Guardado: {archivo}")


def figura_salario_minimo(p: Parametros, archivo: str) -> None:
    K_D, _ = _constantes_demanda_oferta(p)
    eq0 = equilibrio_salario_minimo(p.t0, p)
    eq1 = equilibrio_salario_minimo(p.t1, p)
    eq0_libre = equilibrio_competitivo(p.t0, p)

    L_lo = min(eq0["L"], eq1["L"]) * 0.6
    L_hi = max(eq0["L"], eq1["L"], eq0_libre["L"]) * 1.4
    L_grid = np.linspace(L_lo, L_hi, 400)
    w_hi = max(eq0["w"], eq1["w"], eq0_libre["w"]) * 1.6

    fig, ax = plt.subplots()
    ax.plot(L_grid, _w_demanda(L_grid, p.t0, p, K_D), color="#1f4e79", lw=2,
            label=r"Demanda, $w(1+t_0)$ — antes")
    ax.plot(L_grid, _w_demanda(L_grid, p.t1, p, K_D), color="#1f4e79", lw=2, ls="--",
            label=r"Demanda, $w(1+t_1)$ — después (2013)")
    ax.axhline(eq0["w"], color="#c0392b", lw=2, label=r"Salario mínimo, $\bar{w}$")

    ax.plot([eq0["L"]], [eq0["w"]], "o", color="black", ms=7, zorder=5)
    ax.plot([eq1["L"]], [eq1["w"]], "s", color="black", ms=7, zorder=5)
    ax.plot([eq0_libre["L"]], [eq0_libre["w"]], "x", color="gray", ms=8, zorder=5,
            label="salario competitivo (no restringido)")
    ax.annotate("antes", (eq0["L"], eq0["w"]), textcoords="offset points", xytext=(-30, 8), fontsize=9)
    ax.annotate("después", (eq1["L"], eq1["w"]), textcoords="offset points", xytext=(4, 8), fontsize=9)

    ax.set_xlim(L_lo, L_hi)
    ax.set_ylim(0, w_hi)
    ax.set_xlabel("Empleo, $L$")
    ax.set_ylabel("Salario, $w$")
    ax.set_title("Salario mínimo vinculante: todo el ajuste es en el empleo")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(archivo, dpi=200)
    plt.close(fig)
    print(f"Guardado: {archivo}")


def figura_informal(p: Parametros, archivo: str) -> None:
    K_Df, K_Di = _constantes_informal(p)
    eq0 = equilibrio_informal(p.t0, p)
    eq1 = equilibrio_informal(p.t1, p)
    L_m = p.L0

    Lf_lo = min(eq0["Lf"], eq1["Lf"]) * 0.6
    Lf_hi = L_m - max(eq0["Li"], eq1["Li"]) * 0.6  # deja margen simétrico del lado informal
    Lf_grid = np.linspace(Lf_lo, Lf_hi, 400)
    w_hi = max(eq0["w"], eq1["w"]) * 1.8

    def w_formal(Lf, t):
        return (Lf / K_Df) ** (1 / p.eta) / (1 + t)

    def w_informal(Lf):
        Li = L_m - Lf
        return (Li / K_Di) ** (1 / p.eta_informal)

    fig, ax = plt.subplots()
    ax.plot(Lf_grid, w_formal(Lf_grid, p.t0), color="#1f4e79", lw=2,
            label=r"Demanda formal, $D_f(w(1+t_0))$ — antes")
    ax.plot(Lf_grid, w_formal(Lf_grid, p.t1), color="#1f4e79", lw=2, ls="--",
            label=r"Demanda formal, $D_f(w(1+t_1))$ — después (2013)")
    ax.plot(Lf_grid, w_informal(Lf_grid), color="#27ae60", lw=2,
            label=r"Demanda informal, $D_i(w)$ (no paga $t$)")

    ax.plot([eq0["Lf"]], [eq0["w"]], "o", color="black", ms=7, zorder=5)
    ax.plot([eq1["Lf"]], [eq1["w"]], "s", color="black", ms=7, zorder=5)
    ax.annotate("antes", (eq0["Lf"], eq0["w"]), textcoords="offset points", xytext=(-30, 8), fontsize=9)
    ax.annotate("después", (eq1["Lf"], eq1["w"]), textcoords="offset points", xytext=(4, -14), fontsize=9)
    ax.axvline(L_m, color="gray", lw=1, ls=":", alpha=0.6)

    ax.set_xlim(Lf_lo, Lf_hi)
    ax.set_ylim(0, w_hi)
    ax.set_xlabel(r"Empleo formal, $L_f$  (informal $=L_m-L_f$, leído de derecha a izquierda)")
    ax.set_ylabel("Salario, $w$ (igual en los dos sectores)")
    ax.set_title("Sector informal: reasignación de empleo informal → formal")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(archivo, dpi=200)
    plt.close(fig)
    print(f"Guardado: {archivo}")


def generar_figuras(carpeta: str = ".") -> None:
    p = Parametros()
    figura_competitivo(p, f"{carpeta}/fig_tax_wedge_competitivo.pdf")
    figura_salario_minimo(p, f"{carpeta}/fig_tax_wedge_salario_minimo.pdf")
    figura_informal(p, f"{carpeta}/fig_tax_wedge_informal.pdf")


# ---------------------------------------------------------------------------
# 8. Ejecución principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verificar_formulas()
    resumen_calibracion_2013()
    generar_figuras(".")
