#!/usr/bin/env python3
"""Resumen y graficas de resultados.csv (lo lanza run_all.sh; tambien se puede usar a mano).

    python3 graficas.py resultados/resultados.csv resultados

Genera:
  resumen.txt            tiempos, speed-up, eficiencia, fraccion serie (Karp-Flatt), peso de la
                         comunicacion, cota de Amdahl y comprobacion de que todas las versiones
                         dan el mismo resultado
  speedup.png            speed-up frente a procesos (una grafica por tamano) y la recta ideal
  eficiencia.png         eficiencia = speed-up / procesos
  tiempos_knn_mpi.png    reparto del tiempo del paso 1 (carga, ajuste, evaluacion, comunicacion)
  tiempos_versiones.png  tiempo total de cada version (tamano mayor)
Sin matplotlib solo se genera resumen.txt.

Versiones (columna "version" del CSV):
  knn_seq        secuencial (referencia)          knn_mpi_spec   paso 2: funcional + especulativa
  knn_mpi        paso 1: paralelismo de datos     knn_mpi_dyn    paso 3: cola dinamica + ayuda

Definiciones (T = tiempo total medio de las repeticiones):
  speed-up(N, p)  = T_secuencial(N) / T_paralelo(N, p)   (mismo N y mismos argumentos)
  eficiencia      = speed-up / p
  Karp-Flatt      e = (1/S - 1/p) / (1 - 1/p): fraccion serie "efectiva" medida. Si crece con p hay
                  sobrecoste de comunicacion, sincronizacion o desequilibrio; si es constante es
                  la parte realmente secuencial (ley de Amdahl).
  Amdahl          la carga del dataset la repite cada proceso, asi que no se acelera. Con
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
    "knn_mpi": "paso 1: datos",
    "knn_mpi_spec": "paso 2: especulativa",
    "knn_mpi_dyn": "paso 3: dinamica+ayuda",
}
COLORS = {"knn_mpi": "#1f77b4", "knn_mpi_spec": "#d62728", "knn_mpi_dyn": "#2ca02c"}
MPI = ["knn_mpi", "knn_mpi_spec", "knn_mpi_dyn"]


def num(x):
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


rows = list(csv.DictReader(open(csv_path, newline="")))
if not rows:
    sys.exit("resultados.csv esta vacio")

groups = defaultdict(list)  # (version, N, np) -> filas
for r in rows:
    groups[(r["version"], int(r["tuning"]), int(r["np"]))].append(r)


def mean(key, col):
    vals = [num(r[col]) for r in groups[key] if num(r[col]) is not None]
    return st.mean(vals) if vals else None


sizes = sorted({k[1] for k in groups})
versions = [v for v in MPI if any(k[0] == v for k in groups)]


def nps_of(v, n):
    return sorted(k[2] for k in groups if k[0] == v and k[1] == n and mean(k, "total_s"))


tseq = {n: mean(("knn_seq", n, 1), "total_s") for n in sizes if ("knn_seq", n, 1) in groups}

# ------------------------------------------------------------------ resumen
L = []
L.append("RESUMEN  (tiempos en segundos; media de las repeticiones)")
L.append("speed-up = T_secuencial / T_paralelo (mismo N);  eficiencia = speed-up / procesos;  KF = Karp-Flatt (fraccion serie efectiva)")
L.append("")
for n in sizes:
    any_row = next(r for k, rs in groups.items() if k[1] == n for r in rs)
    L.append(f"=== N = {n}  (test = {any_row['test']}, entrenamiento final = {any_row['final_train']})")
    t0 = tseq.get(n)
    if t0 is None:
        L.append("   (sin medida secuencial: no se pueden calcular speed-up ni eficiencia)")
    else:
        load = mean(("knn_seq", n, 1), "load_s") or 0.0
        tune = mean(("knn_seq", n, 1), "tune_s") or 0.0
        fin = mean(("knn_seq", n, 1), "final_s") or 0.0
        L.append(f"   secuencial: total {t0:.3f}  (carga {load:.3f}, ajuste {tune:.3f} = {100 * tune / t0:.0f} %, "
                 f"evaluacion final {fin:.3f} = {100 * fin / t0:.0f} %)")
        if load > 0:
            f = load / t0
            L.append(f"   Amdahl: fraccion secuencial medida (carga replicada) f = {f:.4f}  ->  speed-up maximo teorico <= {1 / f:.1f}")
    L.append(f"   {'version':<24}{'p':>4}{'total':>10}{'speed-up':>10}{'efic.':>7}{'KF':>8}   notas")
    for v in versions:
        for p in nps_of(v, n):
            t = mean((v, n, p), "total_s")
            s = (t0 / t) if t0 else None
            e = (s / p) if s else None
            kf = ((1 / s - 1 / p) / (1 - 1 / p)) if (s and p > 1) else None
            note = ""
            if v == "knn_mpi":
                comm = mean((v, n, p), "comm_s")
                if comm is not None:
                    note = f"comunicacion+espera={comm:.3f}s ({100 * comm / t:.1f} % del total)"
            else:
                nt, ne = mean((v, n, p), "nT"), mean((v, n, p), "nE")
                rs, ws = mean((v, n, p), "restarts"), mean((v, n, p), "wasted_s")
                if None not in (nt, ne, rs, ws):
                    note = f"ajuste={int(nt)} eval={int(ne)} reinicios={rs:.1f} descartado={ws:.2f}s"
            f2 = lambda x, d=2: (f"{x:.{d}f}" if x is not None else "-")  # noqa: E731
            L.append(f"   {NAMES[v]:<24}{p:>4}{t:>10.3f}{f2(s):>10}{f2(e):>7}{f2(kf, 3):>8}   {note}")
    L.append("")

L.append("COMPROBACION DE CONSISTENCIA (mejor K y accuracy de test, por tamano; deben ser iguales en TODAS las versiones)")
for n in sizes:
    combos = {(r["best_k"], r["test_acc"]) for k, rs in groups.items() if k[1] == n for r in rs}
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
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.8 * len(sizes), 4.1), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        t0 = tseq.get(n)
        if t0 is None:
            ax.set_title(f"N={n} (sin secuencial)")
            continue
        allp = set()
        for v in versions:
            ps = nps_of(v, n)
            if ps:
                ax.plot(ps, [metric(t0 / mean((v, n, p), "total_s"), p) for p in ps], marker="o",
                        label=NAMES[v], color=COLORS[v])
                allp.update(ps)
        if allp:
            ps = sorted(allp)
            ax.plot(ps, [ideal(p) for p in ps], "k:", linewidth=1, label="ideal")
            ax.set_xticks(ps)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("procesos MPI")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)


if tseq and versions:
    panel_figure(lambda s, p: s, "speed-up", "speedup.png", lambda p: p)
    panel_figure(lambda s, p: s / p, "eficiencia", "eficiencia.png", lambda p: 1.0)

# reparto de tiempo del paso 1 (tamano mayor)
if "knn_mpi" in versions:
    n = max(n for n in sizes if nps_of("knn_mpi", n))
    ps = nps_of("knn_mpi", n)
    parts = [
        ("carga (replicada)", "load_s", "#9aa5b1"),
        ("ajuste: calculo", "tune_s", "#1f77b4"),
        ("evaluacion final: calculo", "final_s", "#2ca02c"),
        ("comunicacion + espera", "comm_s", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bottom = [0.0] * len(ps)
    for lab, col, color in parts:
        vals = [mean(("knn_mpi", n, p), col) or 0.0 for p in ps]
        ax.bar([str(p) for p in ps], vals, bottom=bottom, label=lab, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title(f"Paso 1: reparto del tiempo (N = {n}); cada fase = maximo entre procesos")
    ax.title.set_fontsize(9)
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("segundos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tiempos_knn_mpi.png"), dpi=130)
    plt.close(fig)

# tiempo total de cada version (tamano mayor)
if versions:
    n = max(sizes)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for v in versions:
        ps = nps_of(v, n)
        if ps:
            ax.plot(ps, [mean((v, n, p), "total_s") for p in ps], marker="o", label=NAMES[v], color=COLORS[v])
    if tseq.get(n):
        ax.axhline(tseq[n], color="gray", linestyle=":", linewidth=1, label="secuencial")
    ax.set_title(f"Tiempo total (N = {n})")
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("segundos")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tiempos_versiones.png"), dpi=130)
    plt.close(fig)

print("Graficas guardadas en", out_dir)
