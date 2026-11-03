#!/usr/bin/env python3
"""Resumen y graficas de resultados.csv (lo lanza run_all.sh; tambien se puede usar a mano).

    python3 graficas.py resultados/resultados.csv resultados

Genera:
  resumen.txt            tiempos, speed-up, eficiencia, fraccion serie (Karp-Flatt), peso de la
                         comunicacion, cota de Amdahl, efecto de las limpiezas de la Tarea 0 y
                         comprobacion de que todas las versiones dan el mismo resultado
  speedup.png            speed-up frente a procesos (una grafica por tamano) y la recta ideal
  eficiencia.png         eficiencia = speed-up / procesos
  tiempos_knn_mpi.png    reparto del tiempo del paso 1 (carga, ajuste, evaluacion, comunicacion)
  tiempos_versiones.png  tiempo total de cada version (tamano mayor)
Sin matplotlib solo se genera resumen.txt.

Versiones (columna "version" del CSV; el sufijo _opt = biblioteca con las limpiezas de la Tarea 0):
  knn_seq        secuencial                      knn_mpi_spec   paso 2: funcional + especulativa
  knn_mpi        paso 1: paralelismo de datos    knn_mpi_dyn    paso 3: cola dinamica + ayuda

Definiciones (T = tiempo total medio de las repeticiones):
  speed-up(N, p)   = T_secuencial(N) / T_paralelo(N, p)  (mismo N y mismos argumentos). La linea base
                     es el secuencial del MISMO sabor (orig u opt): asi el speed-up mide solo la
                     paralelizacion. La columna "vs orig" usa siempre el secuencial original
                     (ganancia total = limpiezas + paralelismo).
  eficiencia       = speed-up / p
  Karp-Flatt       e = (1/S - 1/p) / (1 - 1/p): fraccion serie "efectiva" medida; si crece con p hay
                     sobrecoste de comunicacion/sincronizacion/desequilibrio, si es constante es
                     la parte realmente secuencial (ley de Amdahl).
  Amdahl           la carga del dataset la repite cada proceso, asi que no se acelera. Con
                     f = carga / T_secuencial, el speed-up maximo teorico es 1 / f. Es una cota
                     MINIMA de la fraccion secuencial (hay mas partes no paralelizables).
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    sys.exit("Uso: graficas.py resultados.csv carpeta_salida")
csv_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

NAMES = {
    "knn_seq": "secuencial",
    "knn_mpi": "paso 1: datos",
    "knn_mpi_spec": "paso 2: especulativa",
    "knn_mpi_dyn": "paso 3: dinamica+ayuda",
}
COLORS = {"knn_mpi": "#1f77b4", "knn_mpi_spec": "#d62728", "knn_mpi_dyn": "#2ca02c"}
MPI_BASES = ["knn_mpi", "knn_mpi_spec", "knn_mpi_dyn"]


def num(x):
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


def split_version(v):
    """'knn_mpi_dyn_opt' -> ('knn_mpi_dyn', 'opt'); 'knn_seq' -> ('knn_seq', 'orig')."""
    return (v[:-4], "opt") if v.endswith("_opt") else (v, "orig")


rows = list(csv.DictReader(open(csv_path, newline="")))
if not rows:
    sys.exit("resultados.csv esta vacio")

# (base, sabor, N, np) -> filas
groups = defaultdict(list)
for r in rows:
    base, fl = split_version(r["version"])
    groups[(base, fl, int(r["tuning"]), int(r["np"]))].append(r)


def mean(key, col):
    vals = [num(r[col]) for r in groups[key] if num(r[col]) is not None]
    return st.mean(vals) if vals else None


sizes = sorted({k[2] for k in groups})
flavors = sorted({k[1] for k in groups}, key=lambda f: (f != "orig", f))
mpi_keys = sorted({(k[0], k[1]) for k in groups if k[0] in MPI_BASES},
                  key=lambda t: (MPI_BASES.index(t[0]), t[1] != "orig"))


def nps_of(base, fl, n):
    return sorted(k[3] for k in groups if k[0] == base and k[1] == fl and k[2] == n)


def seq_time(fl, n):
    return mean(("knn_seq", fl, n, 1), "total_s") if ("knn_seq", fl, n, 1) in groups else None


def baseline(fl, n):
    """Secuencial de referencia para un sabor: el suyo; si no hay, el original; si no, el otro."""
    for f in (fl, "orig", "opt"):
        t = seq_time(f, n)
        if t:
            return t, f
    return None, None


def label(base, fl):
    return NAMES[base] + (" [opt]" if fl == "opt" else (" [orig]" if "opt" in flavors else ""))


# ------------------------------------------------------------------ resumen
L = []
L.append("RESUMEN  (tiempos en segundos; media de las repeticiones)")
L.append("speed-up = T_secuencial / T_paralelo (mismo N, secuencial del mismo sabor);  eficiencia = speed-up / procesos")
L.append("vs orig = speed-up respecto al secuencial ORIGINAL (incluye el efecto de las limpiezas);  KF = Karp-Flatt")
L.append("sabor [orig] = biblioteca del profesor tal cual;  [opt] = con las limpiezas de la Tarea 0")
L.append("")
for n in sizes:
    any_row = next(r for k, rs in groups.items() if k[2] == n for r in rs)
    L.append(f"=== N = {n}  (test = {any_row['test']}, entrenamiento final = {any_row['final_train']})")
    for fl in flavors:
        t0 = seq_time(fl, n)
        if t0 is None:
            continue
        load = mean(("knn_seq", fl, n, 1), "load_s") or 0.0
        tune = mean(("knn_seq", fl, n, 1), "tune_s") or 0.0
        fin = mean(("knn_seq", fl, n, 1), "final_s") or 0.0
        L.append(f"   secuencial [{fl}]: total {t0:.3f}  (carga {load:.3f}, ajuste {tune:.3f} = {100 * tune / t0:.0f} %, "
                 f"evaluacion final {fin:.3f} = {100 * fin / t0:.0f} %)")
        if load > 0:
            f = load / t0
            L.append(f"      Amdahl: fraccion secuencial medida (carga replicada) f = {f:.4f}  ->  speed-up maximo teorico <= {1 / f:.1f}")
    if seq_time("orig", n) and seq_time("opt", n):
        L.append(f"   efecto de las limpiezas en el secuencial: {seq_time('orig', n):.3f} s -> {seq_time('opt', n):.3f} s "
                 f"(x{seq_time('orig', n) / seq_time('opt', n):.3f})")
    L.append(f"   {'version':<30}{'p':>4}{'total':>10}{'speed-up':>10}{'efic.':>7}{'KF':>8}{'vs orig':>9}   notas")
    t_orig = seq_time("orig", n)
    for base, fl in mpi_keys:
        for p in nps_of(base, fl, n):
            t = mean((base, fl, n, p), "total_s")
            if not t:
                continue
            t0, _ = baseline(fl, n)
            s = (t0 / t) if t0 else None
            e = (s / p) if s else None
            kf = ((1 / s - 1 / p) / (1 - 1 / p)) if (s and p > 1) else None
            so = (t_orig / t) if t_orig else None
            note = ""
            if base == "knn_mpi":
                comm = mean((base, fl, n, p), "comm_s")
                if comm is not None:
                    note = f"comunicacion+espera={comm:.3f}s ({100 * comm / t:.1f} % del total)"
            else:
                nt, ne = mean((base, fl, n, p), "nT"), mean((base, fl, n, p), "nE")
                rs, ws = mean((base, fl, n, p), "restarts"), mean((base, fl, n, p), "wasted_s")
                if None not in (nt, ne, rs, ws):
                    note = f"ajuste={int(nt)} eval={int(ne)} reinicios={rs:.1f} descartado={ws:.2f}s"
            f2 = lambda x, d=2: (f"{x:.{d}f}" if x is not None else "-")  # noqa: E731
            L.append(f"   {label(base, fl):<30}{p:>4}{t:>10.3f}{f2(s):>10}{f2(e):>7}{f2(kf, 3):>8}{f2(so):>9}   {note}")
    L.append("")

L.append("COMPROBACION DE CONSISTENCIA (mejor K y accuracy de test, por tamano; deben ser iguales en TODAS las versiones)")
for n in sizes:
    combos = {(r["best_k"], r["test_acc"]) for k, rs in groups.items() if k[2] == n for r in rs}
    state = "OK" if len(combos) == 1 else "AVISO: resultados distintos entre ejecuciones -> " + str(sorted(combos))
    L.append(f"   N={n}: {state}  {sorted(combos)[0] if len(combos) == 1 else ''}")
text = "\n".join(L) + "\n"
with open(os.path.join(out_dir, "resumen.txt"), "w", encoding="utf-8") as fh:
    fh.write(text)
print(text)

# ------------------------------------------------------------------ graficas
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # noqa: BLE001
    print(f"(matplotlib no disponible: {exc}; solo se ha generado resumen.txt)")
    sys.exit(0)


def panel_figure(metric, ylabel, fname, ideal):
    cols = len(sizes)
    fig, axes = plt.subplots(1, cols, figsize=(4.8 * cols, 4.1), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        allp = set()
        for base, fl in mpi_keys:
            t0, _ = baseline(fl, n)
            ps = [p for p in nps_of(base, fl, n) if mean((base, fl, n, p), "total_s")]
            if not t0 or not ps:
                continue
            ys = [metric(t0 / mean((base, fl, n, p), "total_s"), p) for p in ps]
            ax.plot(ps, ys, marker="o", label=label(base, fl), color=COLORS[base],
                    linestyle="-" if fl == "opt" else "--")
            allp.update(ps)
        if allp:
            ps = sorted(allp)
            ax.plot(ps, [ideal(p) for p in ps], "k:", linewidth=1, label="ideal")
        if allp:
            ax.set_xticks(sorted(allp))
        ax.set_title(f"N = {n}")
        ax.set_xlabel("procesos MPI")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)


if mpi_keys and any(seq_time(f, n) for f in flavors for n in sizes):
    panel_figure(lambda s, p: s, "speed-up", "speedup.png", lambda p: p)
    panel_figure(lambda s, p: s / p, "eficiencia", "eficiencia.png", lambda p: 1.0)

# reparto de tiempo del paso 1 (tamano mayor, sabor opt si existe)
cand = [(fl, n) for (b, fl) in mpi_keys if b == "knn_mpi" for n in sizes if nps_of("knn_mpi", fl, n)]
if cand:
    fl, n = max(cand, key=lambda t: (t[1], t[0] == "opt"))
    ps = nps_of("knn_mpi", fl, n)
    parts = [
        ("carga (replicada)", "load_s", "#9aa5b1"),
        ("ajuste: calculo", "tune_s", "#1f77b4"),
        ("evaluacion final: calculo", "final_s", "#2ca02c"),
        ("comunicacion + espera", "comm_s", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bottom = [0.0] * len(ps)
    for lab, col, color in parts:
        vals = [mean(("knn_mpi", fl, n, p), col) or 0.0 for p in ps]
        ax.bar([str(p) for p in ps], vals, bottom=bottom, label=lab, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title(f"Paso 1: reparto del tiempo (N = {n}, {fl}); cada fase = maximo entre procesos")
    ax.title.set_fontsize(9)
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("segundos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tiempos_knn_mpi.png"), dpi=130)
    plt.close(fig)

# tiempo total de cada version (tamano mayor)
if mpi_keys:
    n = max(sizes)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for base, fl in mpi_keys:
        ps = [p for p in nps_of(base, fl, n) if mean((base, fl, n, p), "total_s")]
        if ps:
            ax.plot(ps, [mean((base, fl, n, p), "total_s") for p in ps], marker="o", label=label(base, fl),
                    color=COLORS[base], linestyle="-" if fl == "opt" else "--")
    for fl in flavors:
        t0 = seq_time(fl, n)
        if t0:
            ax.axhline(t0, color="gray", linestyle=":" if fl == "opt" else "-.", linewidth=1, label=f"secuencial [{fl}]")
    ax.set_title(f"Tiempo total (N = {n})")
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("segundos")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tiempos_versiones.png"), dpi=130)
    plt.close(fig)

print("Graficas guardadas en", out_dir)
