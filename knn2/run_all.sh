#!/usr/bin/env bash
grep -q "$(printf '\r')" "$0" 2>/dev/null && sed -i 's/\r$//' "$0" && exec bash "$0" "$@"
# =============================================================================
# run_all.sh  -  Practica 3, Tarea 4 (kNN con MPI)   -   TODO EN UN SOLO SCRIPT
#
# Copia este fichero a la carpeta del proyecto del profesor (la que tiene knn.cc, knn.h,
# crossValidation.cc, mnist.cc, sample.cc, timer.h, get_dataset.sh ...) y ejecutalo:
#
#     chmod +x run_all.sh
#     ./run_all.sh                       # todas las etapas con los valores por defecto (rapido)
#     HOSTFILE=hosts.txt SIZES="4000 8000 20000" NPS="1 2 4 8 12 16" REPS=3 ./run_all.sh
#
# El script lleva DENTRO todo lo que hace falta (al final del fichero): los programas
#   knn_seq.cpp, knn_mpi.cpp, knn_mpi_spec.cpp (pasos 2 y 3), la biblioteca "limpia" de la
#   Tarea 0, graficas.py, los programas de la Tarea 0 y los grafos de flujo (.dot).
# Los extrae en mpi_src/ (no toca los ficheros del profesor), compila, verifica, perfila con
# gprof, mide, dibuja y deja todo en resultados/ (cambiable con OUT=...).
#
# Si el fichero viene de Windows (saltos de linea CRLF): 'bash run_all.sh' se arregla solo; si al
# ejecutar './run_all.sh' sale "bash\r: No existe el fichero", haz una vez:  sed -i 's/\r$//' run_all.sh
#
# ETAPAS (en este orden; sin argumentos se ejecutan todas):
#   arch        arquitectura.txt: procesador, nucleos, caches, memoria (de esta maquina y de cada nodo)
#   extract     extrae los fuentes embebidos a mpi_src/
#   build       comprueba herramientas y dataset, compila TODO en build/ (sabores orig y opt)
#   distribute  copia build/ al resto de nodos del HOSTFILE (ssh sin contrasena)
#   nodos       nodos.txt: donde se lanzan los procesos y si tienen binarios y dataset
#   verify      verificacion.txt: todas las versiones dan EXACTAMENTE lo mismo que el secuencial?
#   gprof       gprof/: perfiles del secuencial (original y limpio) con -O0 y -O2
#   bench       resultados.csv + logs/: tiempos de cada version, tamano, nº de procesos y repeticion
#   plots       resumen.txt, speedup.png, eficiencia.png, ... (speed-up, eficiencia, Amdahl, ...)
#   tarea0      tarea0/: programas del enunciado (vector y matriz) con -O0/-O2 + cambios en la biblioteca
#   grafos      grafos/: grafos de flujo en DOT (+ PNG/SVG si hay graphviz)
#   Ejemplos:  ./run_all.sh bench plots      ./run_all.sh build verify      ./run_all.sh gprof
#
# VARIABLES (todas opcionales)
#   SIZES="2000 4000 8000"  tamanos N del ajuste. Para cada N: test = N/2 y entrenamiento final = 3N
#                           (mantiene la proporcion ~84 % ajuste / 16 % final del problema real, que
#                           es N=20000 -> 10000 test y 60000 entrenamiento). Con N=20000 el
#                           secuencial tarda ~20 min en un PC normal: usalo con REPS=1 o 2.
#   NPS="1 2 4 8"           numeros de procesos MPI. Por defecto: automatico segun los huecos
#                           (slots=) del HOSTFILE o los nucleos de esta maquina (1 2 3 4 6 8 12 ...).
#   REPS=1                  repeticiones de cada medida (recomendado 3 para los resultados finales)
#   HOSTFILE=hosts.txt      fichero de hosts de MPI (una linea por ordenador: "nombre slots=4")
#   MPI_EXTRA=""            opciones extra de mpirun (p. ej. "--map-by node --mca btl_tcp_if_include eth0")
#   SEQ_FLAVORS="orig opt"  secuenciales a medir como linea base (orig = codigo del profesor tal cual,
#                           opt = con las limpiezas de la Tarea 0)
#   MPI_FLAVORS="opt"       sabores MPI a medir en bench ("orig opt" para medir los dos). La
#                           verificacion comprueba siempre los dos.
#   BENCH_VERSIONS="knn_mpi knn_mpi_spec knn_mpi_dyn"   que versiones medir (paso 1, paso 2, paso 3)
#   PROF_N=2000             tamano N con el que se pasa gprof
#   VERIFY_N=1000  VERIFY_NPS="1 2 3 4"   tamano y procesos de la verificacion
#   MAXK=10 FOLDS=5         rango de K y numero de folds
#   OUT=resultados          carpeta de salida;   APPEND=1  anade a resultados.csv en vez de reescribirlo
#   RUN_TIMEOUT=0           segundos maximos por ejecucion (0 = sin limite; util para no quedarse
#                           colgado si un nodo falla)
#   T0_REPS=3  T0_VECTOR="100000 1000000 10000000"  T0_MATRIX="500 1000 2000 4000"   (etapa tarea0)
#   BUILD=<dir>             carpeta de binarios y dataset (por defecto ./build, como en el proyecto)
# =============================================================================

set -u -o pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SELF")" && pwd)"
cd "$ROOT" || exit 1

SIZES=${SIZES:-"2000 4000 8000"}
NPS=${NPS:-}
REPS=${REPS:-1}
PROF_N=${PROF_N:-2000}
VERIFY_N=${VERIFY_N:-1000}
VERIFY_NPS=${VERIFY_NPS:-"1 2 3 4"}
MAXK=${MAXK:-10}
FOLDS=${FOLDS:-5}
HOSTFILE=${HOSTFILE:-}
MPI_EXTRA=${MPI_EXTRA:-}
SEQ_FLAVORS=${SEQ_FLAVORS:-"orig opt"}
MPI_FLAVORS=${MPI_FLAVORS:-"opt"}
BENCH_VERSIONS=${BENCH_VERSIONS:-"knn_mpi knn_mpi_spec knn_mpi_dyn"}
APPEND=${APPEND:-0}
OUT=${OUT:-resultados}
RUN_TIMEOUT=${RUN_TIMEOUT:-0}
T0_REPS=${T0_REPS:-3}
T0_VECTOR=${T0_VECTOR:-"100000 1000000 10000000"}
T0_MATRIX=${T0_MATRIX:-"500 1000 2000 4000"}

case "$OUT" in /*) OUT_DIR="$OUT" ;; *) OUT_DIR="$ROOT/$OUT" ;; esac
BUILD=${BUILD:-$ROOT/build}
case "$BUILD" in /*) ;; *) BUILD="$ROOT/$BUILD" ;; esac
MY="$ROOT/mpi_src"          # fuentes propios extraidos de este script
if [[ -n "$HOSTFILE" ]]; then
    HOSTFILE="$(readlink -f "$HOSTFILE")"
    [[ -f "$HOSTFILE" ]] || { echo "[ERROR] No existe el HOSTFILE '$HOSTFILE'" >&2; exit 1; }
fi
# Dentro de contenedores se suele ser root: Open MPI lo rechaza salvo que se confirme.
if [[ "$(id -u)" == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi

CSV_HEADER="version,rep,np,tuning,test,final_train,max_k,folds,best_k,test_acc,total_s,load_s,tune_s,final_s,comm_s,nT,nE,restarts,wasted_s"

log()  { printf '\n== %s ==\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   [AVISO] %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

hosts_list() { awk '!/^[[:space:]]*(#|$)/ {print $1}' "$HOSTFILE" | sort -u; }
sfx() { [[ "$1" == orig ]] && echo "" || echo "_$1"; }        # sabor -> sufijo del binario

# Tamanos derivados de N (con los topes del dataset MNIST: 60000 train, 10000 test).
test_of()  { local t=$(( $1 / 2 )); (( t > 10000 )) && t=10000; echo "$t"; }
final_of() { local f=$(( 3 * $1 )); (( f > 60000 )) && f=60000; echo "$f"; }

# Huecos de procesos disponibles: suma de slots= del HOSTFILE o nucleos de esta maquina.
slots_total() {
    if [[ -n "$HOSTFILE" ]]; then
        awk '!/^[[:space:]]*(#|$)/ { n=1; for (i=2;i<=NF;i++) if ($i ~ /^slots=/) { split($i,a,"="); n=a[2] } s+=n } END { print s+0 }' "$HOSTFILE"
    else
        nproc 2>/dev/null || echo 4
    fi
}
default_nps() {
    local max c out=""
    max=$(slots_total); (( max >= 1 )) || max=1
    for c in 1 2 3 4 6 8 12 16 24 32 48 64; do (( c <= max )) && out+="$c "; done
    [[ " $out " == *" $max "* ]] || out+="$max"
    echo $out
}
[[ -n "$NPS" ]] || NPS=$(default_nps)

with_timeout() { if (( RUN_TIMEOUT > 0 )); then timeout -k 10 "$RUN_TIMEOUT" "$@"; else "$@"; fi; }

# Ejecuta un programa SECUENCIAL desde build/ (rutas relativas train/ y test/) y quita
# los contadores de progreso "i/N" que imprime KNN::classifyAndEvaluate.
run_seq() {
    local prog=$1; shift
    (cd "$BUILD" && with_timeout "./$prog" "$@") 2>&1 | tr '\r' '\n' | grep -vE '^[0-9]+/[0-9]+$'
}

# Ejecuta un programa MPI desde build/. Con HOSTFILE usa los nodos del fichero.
run_mpi() {
    local np=$1; shift
    local args=(--oversubscribe)
    if [[ -n "$HOSTFILE" ]]; then
        args+=(-mca plm_rsh_no_tree_spawn 1 --hostfile "$HOSTFILE")
    fi
    # shellcheck disable=SC2086
    (cd "$BUILD" && with_timeout mpirun "${args[@]}" $MPI_EXTRA -np "$np" "$@")
}

# launch <version> <np> <N> [MAXK FOLDS]: lanza una version con el tamano N (test y final derivados).
#   knn_seq | knn_mpi | knn_mpi_spec (paso 2, estatico) | knn_mpi_dyn (paso 3, dinamico); sufijo _opt = limpio
launch() {
    local ver=$1 np=$2 N=$3 fl="" T F
    T=$(test_of "$N"); F=$(final_of "$N")
    if [[ "$ver" == *_opt ]]; then fl=_opt; ver=${ver%_opt}; fi
    case "$ver" in
        knn_seq)      run_seq "knn_seq$fl" "$MAXK" "$FOLDS" "$N" "$T" "$F" ;;
        knn_mpi)      run_mpi "$np" "./knn_mpi$fl" "$MAXK" "$FOLDS" "$N" "$T" "$F" ;;
        knn_mpi_spec) run_mpi "$np" "./knn_mpi_spec$fl" "$MAXK" "$FOLDS" "$N" "$T" "$F" 0 1 0 ;;
        knn_mpi_dyn)  run_mpi "$np" "./knn_mpi_spec$fl" "$MAXK" "$FOLDS" "$N" "$T" "$F" 0 1 1 ;;
        *) die "Version desconocida: $ver" ;;
    esac
}

# Lineas que deben coincidir entre versiones: accuracy por K, mejor K y accuracy de test.
extract_res() {
    grep -E '^(K:|Best k|Test accuracy)' | sed -E 's/ completed in.*//; s/Best k: ([0-9]+).*/Best k: \1/'
}

need_binaries() {
    local b fl
    for fl in $SEQ_FLAVORS; do
        [[ -x "$BUILD/knn_seq$(sfx "$fl")" ]] || die "Falta build/knn_seq$(sfx "$fl"): ejecuta primero './run_all.sh build'"
    done
    for fl in $MPI_FLAVORS; do
        for b in knn_mpi knn_mpi_spec; do
            [[ -x "$BUILD/$b$(sfx "$fl")" ]] || die "Falta build/$b$(sfx "$fl"): ejecuta primero './run_all.sh build'"
        done
    done
}

# Escribe en $2 el fichero embebido $1 (secciones "#@@BEGIN nombre" ... "#@@END nombre" al final de este script).
emit() {
    mkdir -p "$(dirname "$2")"
    awk -v n="$1" '$0 == "#@@BEGIN " n {f=1; next} $0 == "#@@END " n {f=0} f' "$SELF" > "$2"
    [[ -s "$2" ]] || die "No se pudo extraer '$1' del script"
}

# ------------------------------------------------------------------ arch ----
stage_arch() {
    log "Arquitectura del sistema"
    mkdir -p "$OUT_DIR"
    local f="$OUT_DIR/arquitectura.txt"
    {
        echo "Fecha: $(date)"
        echo "Equipo: $(hostname)"
        echo "--- uname -a";  uname -a
        echo "--- lscpu";     lscpu 2>/dev/null || echo "(lscpu no disponible)"
        echo "--- memoria";   free -h 2>/dev/null
        echo "--- caches (getconf)"; getconf -a 2>/dev/null | grep -i cache
        echo "--- herramientas"
        g++ --version 2>/dev/null | head -1
        mpirun --version 2>&1 | head -1
    } > "$f" 2>&1
    if [[ -n "$HOSTFILE" ]]; then
        local h
        for h in $(hosts_list); do
            {
                echo; echo "=================== NODO $h"
                ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
                    'hostname; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz|L1d|L1i|L2|L3"; free -h | head -2' 2>&1
            } >> "$f"
        done
    fi
    info "Guardado en $f"
}

# --------------------------------------------------------------- extract ----
stage_extract() {
    log "Extrayendo los fuentes embebidos a $MY"
    local f
    for f in knn_seq.cpp knn_mpi.cpp knn_mpi_spec.cpp graficas.py \
             opt/knn.h opt/knn.cc opt/crossValidation.h opt/crossValidation.cc \
             tarea0/tarea0_vector.cpp tarea0/tarea0_matriz.cpp tarea0/cambios_tarea0.txt \
             grafos/secuencial.dot grafos/paso1_datos.dot grafos/paso2_3_especulativo.dot; do
        emit "$f" "$MY/$f"
    done
    info "Hecho (los ficheros del profesor no se tocan)"
}

# ----------------------------------------------------------------- build ----
stage_build() {
    log "Herramientas y dataset"
    local t
    for t in g++ mpic++ mpirun gprof; do
        command -v "$t" >/dev/null 2>&1 || die "Falta '$t' (instala build-essential y openmpi-bin libopenmpi-dev)"
        info "$t: $(command -v "$t")"
    done
    for t in knn.cc knn.h crossValidation.cc crossValidation.h mnist.cc mnist.h sample.cc sample.h timer.h; do
        [[ -f "$ROOT/$t" ]] || die "Falta $t en $ROOT: copia run_all.sh a la carpeta del proyecto del profesor"
    done
    mkdir -p "$BUILD" "$OUT_DIR"
    stage_extract

    local need=(train/train-images-idx3-ubyte train/train-labels-idx1-ubyte test/t10k-images-idx3-ubyte test/t10k-labels-idx1-ubyte)
    local f missing=0
    for f in "${need[@]}"; do [[ -s "$BUILD/$f" ]] || missing=1; done
    if (( missing )); then
        info "Falta el dataset en build/: ejecutando get_dataset.sh (necesita wget y red)"
        if [[ -f "$ROOT/get_dataset.sh" ]]; then
            chmod +x "$ROOT/get_dataset.sh"
            (cd "$ROOT" && ./get_dataset.sh) > "$OUT_DIR/get_dataset.log" 2>&1 || warn "get_dataset.sh termino con error (ver get_dataset.log); se comprueba el dataset igualmente"
        fi
        missing=0
        for f in "${need[@]}"; do [[ -s "$BUILD/$f" ]] || missing=1; done
        (( missing )) && die "Sigue faltando el dataset. Copia los 4 archivos IDX a $BUILD/train y $BUILD/test."
    fi
    info "Dataset MNIST: OK"

    log "Compilacion (salida en $OUT_DIR/compilacion.log)"
    local cl="$OUT_DIR/compilacion.log"
    : > "$cl"
    compile() {  # compile <comando...>
        echo "\$ $*" >> "$cl"
        "$@" >> "$cl" 2>&1 || die "Error de compilacion: $* (ver $cl)"
    }
    local fl s inc def lib_seq lib_mpi
    for fl in orig opt; do
        s=$(sfx "$fl")
        if [[ "$fl" == orig ]]; then   # biblioteca del profesor tal cual
            inc=(-I"$ROOT")
            lib_seq=("$ROOT/crossValidation.cc" "$ROOT/knn.cc")
            lib_mpi=("$ROOT/knn.cc")
        else                           # biblioteca con las limpiezas de la Tarea 0 (tiene prioridad en -I)
            inc=(-I"$MY/opt" -I"$ROOT")
            lib_seq=("$MY/opt/crossValidation.cc" "$MY/opt/knn.cc")
            lib_mpi=("$MY/opt/knn.cc")
        fi
        def=(-DFLAVOR="\"$s\"")
        compile g++ -O2 -std=c++17 "${inc[@]}" "${def[@]}" "$MY/knn_seq.cpp" "${lib_seq[@]}" "$ROOT/mnist.cc" "$ROOT/sample.cc" -o "$BUILD/knn_seq$s"
        compile g++ -O0 -g -pg -std=c++17 "${inc[@]}" "${def[@]}" "$MY/knn_seq.cpp" "${lib_seq[@]}" "$ROOT/mnist.cc" "$ROOT/sample.cc" -o "$BUILD/knn_seq${s}_pg_O0"
        compile g++ -O2 -g -pg -std=c++17 "${inc[@]}" "${def[@]}" "$MY/knn_seq.cpp" "${lib_seq[@]}" "$ROOT/mnist.cc" "$ROOT/sample.cc" -o "$BUILD/knn_seq${s}_pg_O2"
        compile mpic++ -O2 -std=c++17 "${inc[@]}" "${def[@]}" "$MY/knn_mpi.cpp" "${lib_mpi[@]}" "$ROOT/mnist.cc" "$ROOT/sample.cc" -o "$BUILD/knn_mpi$s"
        compile mpic++ -O2 -std=c++17 "${inc[@]}" "${def[@]}" "$MY/knn_mpi_spec.cpp" "${lib_mpi[@]}" "$ROOT/mnist.cc" "$ROOT/sample.cc" -o "$BUILD/knn_mpi_spec$s"
        info "Sabor '$fl': knn_seq$s (+ _pg_O0, _pg_O2), knn_mpi$s, knn_mpi_spec$s (modos estatico y dinamico)"
    done
}

# ------------------------------------------------------------ distribute ----
stage_distribute() {
    [[ -n "$HOSTFILE" ]] || { log "Distribucion"; info "Sin HOSTFILE: nada que copiar"; return; }
    log "Copia de build/ al resto de nodos (misma ruta en todos)"
    local me h qdir
    me=$(hostname)
    qdir=$(printf '%q' "$BUILD")
    for h in $(hosts_list); do
        [[ "$h" == "$me" || "$h" == "localhost" || "$h" == "127.0.0.1" ]] && continue
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "mkdir -p $qdir"; then
            warn "No se pudo conectar con $h por ssh (hacen falta claves sin contrasena y el mismo usuario)"
            continue
        fi
        if command -v rsync >/dev/null 2>&1 && ssh -o BatchMode=yes "$h" 'command -v rsync' >/dev/null 2>&1; then
            rsync -a --exclude '*_pg_*' --exclude gmon.out "$BUILD/" "$h:$qdir/" && info "$h: copiado (rsync)" || warn "$h: fallo rsync"
        else
            scp -q -r "$BUILD/." "$h:$qdir/" && info "$h: copiado (scp)" || warn "$h: fallo scp"
        fi
    done
    info "Recuerda: los binarios deben ser validos en todos los nodos (misma arquitectura y librerias MPI)."
    info "Si la carpeta del proyecto esta en un disco compartido (NFS), no hace falta copiar nada."
}

# ----------------------------------------------------------------- nodos ----
stage_nodos() {
    mkdir -p "$OUT_DIR"
    log "Nodos y reparto de procesos"
    local maxnp=1 np
    for np in $VERIFY_NPS $NPS; do (( np > maxnp )) && maxnp=$np; done
    {
        echo "# Un proceso por linea (np=$maxnp): nodo, directorio de trabajo, archivos de train/test y si existe el binario"
        run_mpi "$maxnp" sh -c 'echo "$(hostname) cwd=$(pwd) train=$(ls train 2>/dev/null | wc -l) test=$(ls test 2>/dev/null | wc -l) knn_mpi_opt=$([ -x ./knn_mpi_opt ] && echo si || echo NO)"' 2>&1 | sort
    } > "$OUT_DIR/nodos.txt"
    sed 's/^/   /' "$OUT_DIR/nodos.txt"
    if grep -qE 'NO$|train=0|test=0' "$OUT_DIR/nodos.txt"; then
        warn "Algun nodo no tiene los binarios o el dataset en build/. Ejecuta la etapa 'distribute'."
    fi
}

# ---------------------------------------------------------------- verify ----
stage_verify() {
    local fl b
    for fl in orig opt; do
        [[ -x "$BUILD/knn_seq$(sfx "$fl")" && -x "$BUILD/knn_mpi$(sfx "$fl")" && -x "$BUILD/knn_mpi_spec$(sfx "$fl")" ]] \
            || die "Faltan binarios del sabor '$fl': ejecuta primero './run_all.sh build'"
    done
    mkdir -p "$OUT_DIR"
    stage_nodos

    local N=$VERIFY_N T F
    T=$(test_of "$N"); F=$(final_of "$N")
    log "Verificacion: N=$N (test=$T, entrenamiento final=$F), procesos: $VERIFY_NPS"
    local ref got ok=1 np ver vlab res
    ref=$(launch knn_seq 1 "$N" | extract_res)
    [[ -n "$ref" ]] || die "El secuencial no produjo resultados"
    {
        echo "Referencia: knn_seq (original, codigo del profesor) con maxK=$MAXK folds=$FOLDS N=$N test=$T entrenamiento_final=$F"
        echo "$ref"
        echo
    } > "$OUT_DIR/verificacion.txt"

    check() {  # check <etiqueta> <salida obtenida>
        if [[ "$2" == "$ref" ]]; then res="OK"; else
            res="FALLO"; ok=0
            { echo "--- $1: DIFERENCIAS respecto al secuencial"; diff <(echo "$ref") <(echo "$2"); } >> "$OUT_DIR/verificacion.txt"
        fi
        printf '%-26s %s\n' "$1" "$res" | tee -a "$OUT_DIR/verificacion.txt" | sed 's/^/   /'
    }
    check "knn_seq_opt (secuencial limpio)" "$(launch knn_seq_opt 1 "$N" | extract_res)"
    for fl in orig opt; do
        [[ $fl == orig ]] && b="" || b="_opt"
        for np in $VERIFY_NPS; do
            for ver in knn_mpi knn_mpi_spec knn_mpi_dyn; do
                [[ $ver != knn_mpi && $np -lt 2 ]] && continue
                vlab="$ver$b np=$np"
                got=$(launch "$ver$b" "$np" "$N" 2>&1 | extract_res)
                check "$vlab" "$got"
            done
        done
    done
    if (( ok )); then
        echo "RESULTADO: todas las versiones coinciden con el secuencial" | tee -a "$OUT_DIR/verificacion.txt" | sed 's/^/   /'
    else
        echo "RESULTADO: HAY DIFERENCIAS (ver verificacion.txt)" | tee -a "$OUT_DIR/verificacion.txt" | sed 's/^/   /'
        warn "No sigas con las mediciones hasta resolverlo."
    fi
}

# ----------------------------------------------------------------- gprof ----
top_flat() {  # primeras filas del perfil plano de un informe gprof
    awk '/^Flat profile/{f=1;next} /^Call graph/{f=0} f && NF>=4 && $1 ~ /^[0-9]+(\.[0-9]+)?$/ {s=$0; if (length(s)>110) s=substr(s,1,107) "..."; print s; if (++n>=8) exit}' "$1"
}

stage_gprof() {
    local fl s O
    for fl in orig opt; do
        s=$(sfx "$fl")
        [[ -x "$BUILD/knn_seq${s}_pg_O0" && -x "$BUILD/knn_seq${s}_pg_O2" ]] || die "Faltan los binarios con -pg (sabor $fl): ejecuta './run_all.sh build'"
    done
    log "gprof sobre el secuencial (N=$PROF_N, test=$(test_of "$PROF_N"), entrenamiento final=$(final_of "$PROF_N"))"
    local d="$OUT_DIR/gprof"
    mkdir -p "$d"
    for fl in orig opt; do
        s=$(sfx "$fl")
        for O in O0 O2; do
            info "Ejecutando knn_seq${s}_pg_$O ..."
            rm -f "$BUILD/gmon.out"
            run_seq "knn_seq${s}_pg_$O" "$MAXK" "$FOLDS" "$PROF_N" "$(test_of "$PROF_N")" "$(final_of "$PROF_N")" > "$d/salida_${fl}_$O.txt"
            if [[ ! -f "$BUILD/gmon.out" ]]; then warn "No se genero gmon.out ($fl, -$O)"; continue; fi
            mv "$BUILD/gmon.out" "$d/gmon_${fl}_$O.out"
            (cd "$BUILD" && gprof -b -p -q "./knn_seq${s}_pg_$O" "$d/gmon_${fl}_$O.out") > "$d/perfil_${fl}_$O.txt" 2>&1
            info "perfil_${fl}_$O.txt generado"
        done
    done
    {
        echo "Resumen gprof (secuencial, N=$PROF_N). Columnas: %tiempo  acumulado  propio  llamadas  ms/llamada  ms/llamada-total  funcion"
        echo "orig = codigo del profesor; opt = con las limpiezas de la Tarea 0 (ver tarea0/cambios_tarea0.txt)."
        echo "Con -O2 hay funciones que desaparecen por 'inlining' y el cuello de botella puede cambiar de sitio."
        echo "Los informes completos (perfil plano + grafo de llamadas) estan en perfil_<sabor>_<O>.txt."
        for fl in orig opt; do
            for O in O0 O2; do
                echo; echo "=== $fl, -$O: funciones con mas tiempo propio ==="
                [[ -f "$d/perfil_${fl}_$O.txt" ]] && top_flat "$d/perfil_${fl}_$O.txt"
            done
        done
    } > "$d/resumen_gprof.txt"
    sed 's/^/   /' "$d/resumen_gprof.txt"
}

# ----------------------------------------------------------------- bench ----
run_one() {  # run_one <version> <np> <N> <rep>
    local ver=$1 np=$2 N=$3 rep=$4 logf line
    logf="$OUT_DIR/logs/${ver}_N${N}_np${np}_r${rep}.log"
    launch "$ver" "$np" "$N" > "$logf" 2>&1
    line=$(grep '^CSV,' "$logf" | tail -1)
    if [[ -z "$line" ]]; then
        warn "Fallo: $ver N=$N np=$np rep=$rep (ver $logf)"
        return
    fi
    line=${line#CSV,}
    echo "${line%%,*},$rep,${line#*,}" >> "$OUT_DIR/resultados.csv"
    printf '   %-20s N=%-6s np=%-3s rep=%-2s total=%s s\n' "$ver" "$N" "$np" "$rep" "$(echo "$line" | cut -d, -f10)"
}

stage_bench() {
    need_binaries
    mkdir -p "$OUT_DIR/logs"
    if [[ "$APPEND" != 1 || ! -f "$OUT_DIR/resultados.csv" ]]; then
        echo "$CSV_HEADER" > "$OUT_DIR/resultados.csv"
    fi
    log "Mediciones: SIZES='$SIZES' NPS='$NPS' REPS=$REPS SEQ_FLAVORS='$SEQ_FLAVORS' MPI_FLAVORS='$MPI_FLAVORS' ${HOSTFILE:+HOSTFILE=$HOSTFILE}"
    local N rep np fl b ver
    for N in $SIZES; do
        info "--- N=$N (test=$(test_of "$N"), entrenamiento final=$(final_of "$N"))"
        for rep in $(seq 1 "$REPS"); do
            for fl in $SEQ_FLAVORS; do run_one "knn_seq$(sfx "$fl")" 1 "$N" "$rep"; done
            for np in $NPS; do
                for fl in $MPI_FLAVORS; do
                    b=$(sfx "$fl")
                    for ver in $BENCH_VERSIONS; do
                        [[ $ver != knn_mpi && $np -lt 2 ]] && continue   # los pasos 2 y 3 necesitan >= 2 procesos
                        run_one "$ver$b" "$np" "$N" "$rep"
                    done
                done
            done
        done
    done
    info "Tiempos en $OUT_DIR/resultados.csv, salidas completas en $OUT_DIR/logs/"
}

# ----------------------------------------------------------------- plots ----
stage_plots() {
    log "Resumen y graficas"
    [[ -s "$OUT_DIR/resultados.csv" ]] || die "No hay $OUT_DIR/resultados.csv: ejecuta antes la etapa 'bench'"
    command -v python3 >/dev/null 2>&1 || { warn "No hay python3: solo queda el CSV"; return; }
    [[ -f "$MY/graficas.py" ]] || emit graficas.py "$MY/graficas.py"
    python3 "$MY/graficas.py" "$OUT_DIR/resultados.csv" "$OUT_DIR" || warn "graficas.py termino con error"
}

# ---------------------------------------------------------------- tarea0 ----
stage_tarea0() {
    log "Tarea 0: vector (push_back vs tamano fijo) y matriz (columnas vs filas)"
    local d="$OUT_DIR/tarea0" O n r prog
    mkdir -p "$d/bin"
    [[ -f "$MY/tarea0/tarea0_vector.cpp" ]] || stage_extract
    local raw="$d/crudo.csv"
    echo "programa,opt,size,rep,tiempo_A_s,tiempo_B_s,ganancia_A_entre_B" > "$raw"
    for O in O0 O2; do
        for prog in vector matriz; do
            local sizes=$T0_VECTOR; [[ $prog == matriz ]] && sizes=$T0_MATRIX
            for n in $sizes; do
                g++ -$O -DSIZE="$n" "$MY/tarea0/tarea0_$prog.cpp" -o "$d/bin/t0_${prog}_${O}_$n" >> "$d/compilacion.log" 2>&1 \
                    || { warn "No compila tarea0_$prog con -$O SIZE=$n (ver tarea0/compilacion.log)"; continue; }
                for r in $(seq 1 "$T0_REPS"); do
                    "$d/bin/t0_${prog}_${O}_$n" | awk -v p="$prog" -v o="$O" -v n="$n" -v r="$r" \
                        '/Elapsed/ { gsub("s$","",$3); t[++i]=$3 } /Gain/ { g=$2 } END { printf "%s,%s,%d,%d,%s,%s,%s\n", p, o, n, r, t[1], t[2], g }' >> "$raw"
                done
            done
        done
    done
    rm -rf "$d/bin"
    awk -F, 'NR > 1 { k = $1 "," $2 "," $3; a[k] += $5; b[k] += $6; c[k]++ ; if (!(k in ord)) { ord[k] = ++m; key[m] = k } }
             END {
               printf "%-8s %-4s %-10s %14s %14s %10s\n", "programa", "opt", "size", "A (s)", "B (s)", "A / B";
               for (j = 1; j <= m; j++) { k = key[j]; split(k, f, ","); A = a[k] / c[k]; B = b[k] / c[k];
                 printf "%-8s %-4s %-10s %14.6f %14.6f %10.2f\n", f[1], f[2], f[3], A, B, (B > 0 ? A / B : 0) } }' "$raw" > "$d/tabla.txt"
    {
        echo "TAREA 0 - media de $T0_REPS repeticiones (tiempos de CPU, clock())"
        echo "vector: A = push_back sobre un vector vacio;  B = vector creado con el tamano final"
        echo "matriz: A = recorrido por COLUMNAS;           B = recorrido por FILAS"
        echo "A / B > 1 significa que B (reservar el tamano / recorrer por filas) es mas rapido."
        echo
        cat "$d/tabla.txt"
        echo
        echo "Lectura: con -O0 el compilador no optimiza nada y se ve el coste puro de las reasignaciones"
        echo "(vector) y de los fallos de cache (matriz). Con -O2 la mejora del vector suele ser mayor"
        echo "(el bucle B se reduce a un relleno de memoria) y la de la matriz depende de si la matriz cabe"
        echo "en cache: para tamanos pequenos casi no hay diferencia; para tamanos grandes aparece."
    } > "$d/resumen_tarea0.txt"
    {
        cat "$MY/tarea0/cambios_tarea0.txt"
        local f
        for f in knn.h knn.cc crossValidation.h crossValidation.cc; do
            echo; echo "########## diff -u $f (original -> limpio)"
            diff -u "$ROOT/$f" "$MY/opt/$f"
        done
    } > "$d/cambios_tarea0.txt"
    sed 's/^/   /' "$d/resumen_tarea0.txt"
    info "Cambios en la biblioteca: $d/cambios_tarea0.txt"
}

# ---------------------------------------------------------------- grafos ----
stage_grafos() {
    log "Grafos de flujo (DOT)"
    local d="$OUT_DIR/grafos" f g
    mkdir -p "$d"
    for f in secuencial paso1_datos paso2_3_especulativo; do
        emit "grafos/$f.dot" "$d/$f.dot"
        if command -v dot >/dev/null 2>&1; then
            dot -Tpng "$d/$f.dot" -o "$d/$f.png" && dot -Tsvg "$d/$f.dot" -o "$d/$f.svg" && info "$f: .dot, .png y .svg"
        else
            info "$f: solo .dot (instala graphviz: sudo apt install graphviz, o pega el .dot en https://dreampuf.github.io/GraphvizOnline)"
        fi
    done
}

# ----------------------------------------------------------------- leeme ----
write_leeme() {
    cat > "$OUT_DIR/LEEME.txt" <<'EOF'
Contenido de la carpeta de resultados (lo genera run_all.sh)
  arquitectura.txt     procesador, nucleos, caches y memoria (de esta maquina y de cada nodo)  -> memoria
  compilacion.log      ordenes de compilacion y avisos
  nodos.txt            en que nodos arrancan los procesos y si tienen binarios y dataset
  verificacion.txt     todas las versiones dan exactamente lo mismo que el secuencial?
  gprof/               perfil_<sabor>_<O>.txt (informes gprof), resumen_gprof.txt, salida_*.txt, gmon_*.out
                       sabor: orig = codigo del profesor, opt = con las limpiezas de la Tarea 0
  resultados.csv       una fila por ejecucion (version, rep, np, tamanos, K, accuracy, tiempos, ...)
  logs/                la salida completa de cada ejecucion
  resumen.txt          speed-up, eficiencia, Karp-Flatt, comunicacion, cota de Amdahl, efecto de limpiezas
  speedup.png eficiencia.png tiempos_knn_mpi.png tiempos_versiones.png
  tarea0/              resumen_tarea0.txt (vector y matriz), cambios_tarea0.txt (que se cambio en la biblioteca)
  grafos/              grafos de flujo: secuencial, paso 1, pasos 2-3 (.dot, .png, .svg)
Versiones (columna "version" del CSV; sufijo _opt = con limpiezas):
  knn_seq        secuencial                           knn_mpi_spec   paso 2: funcional + especulativa (estatica)
  knn_mpi        paso 1: paralelismo de datos         knn_mpi_dyn    paso 3: cola dinamica + ayuda de evaluadores
EOF
}

# ------------------------------------------------------------------ main ----
ALL_STAGES="arch extract build distribute nodos verify gprof bench plots tarea0 grafos"
REQUESTED=()
for a in "$@"; do a=${a//$'\r'/}; [[ -n "$a" ]] && REQUESTED+=("$a"); done
(( ${#REQUESTED[@]} )) || REQUESTED=(all)

for s in "${REQUESTED[@]}"; do
    [[ "$s" == all ]] && continue
    [[ " $ALL_STAGES " == *" $s "* ]] || die "Etapa desconocida '$s'. Etapas: $ALL_STAGES"
done

wanted() {
    local r
    for r in "${REQUESTED[@]}"; do [[ "$r" == all || "$r" == "$1" ]] && return 0; done
    return 1
}

mkdir -p "$OUT_DIR"
T_START=$SECONDS
for s in $ALL_STAGES; do
    # 'extract' y 'nodos' ya forman parte de 'build' y 'verify'; en "todo" no se repiten.
    if [[ "$s" == extract || "$s" == nodos ]]; then
        [[ " ${REQUESTED[*]} " == *" $s "* ]] || continue
        [[ "$s" == nodos ]] && wanted verify && continue
        "stage_$s"
        continue
    fi
    wanted "$s" && "stage_$s"
done
write_leeme

log "Terminado en $(( SECONDS - T_START )) s. Resultados en $OUT_DIR"
ls -1 "$OUT_DIR" | sed 's/^/   /'
exit 0

# ===================== FICHEROS EMBEBIDOS (no editar a mano; los lee 'emit') =====================
: <<'__PAYLOAD_NO_EJECUTAR__'
#@@BEGIN knn_seq.cpp
// =============================================================================
// knn_seq.cpp  -  Version SECUENCIAL parametrizada
//
// Hace exactamente lo mismo que main.cc (mismas clases CrossValidation y KNN,
// mismo fineTuning, mismas salidas por pantalla). La unica diferencia es que el
// tamano del problema se puede elegir por linea de comandos, con los MISMOS
// argumentos que knn_mpi y knn_mpi_spec:
//
//     ./knn_seq [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]
//     por defecto: 10 5 20000 todas todas   (= el problema de main.cc)
//
// Sirve para dos cosas:
//   * Referencia secuencial AL MISMO TAMANO que las versiones MPI (speed-up).
//   * Perfilar con gprof a un tamano reducido (main.cc fija 20000 muestras).
//
// Compilar:  g++ -O2 -std=c++17 -I. knn_seq.cpp crossValidation.cc mnist.cc knn.cc sample.cc -o build/knn_seq
// =============================================================================

#include "sample.h"
#include "knn.h"
#include "mnist.h"
#include "crossValidation.h"

#include "timer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef FLAVOR
#define FLAVOR "" // run_all.sh compila la variante con la biblioteca limpia con -DFLAVOR='"_opt"'
#endif

using namespace knn;
using namespace mnist;
using namespace crossValidation;

// Copia de fineTuning de main.cc (misma logica y mismas salidas).
int fineTuning(unsigned int minK, unsigned int maxK, const std::vector<TSample> &dataset, unsigned int folds)
{
    assert(minK >= 1 && maxK > minK);

    CrossValidation crossValidation = CrossValidation::NFold(dataset, folds);
    std::cout<<"Starting search for best K parameter in range ["<<minK<<", "<<maxK<<"] using "<<crossValidation.getName()<<" with "<<dataset.size()<<" samples."<<std::endl;

    unsigned int bestK = minK;
    double bestAccuracy = 0.0;
    for(int k = minK; k <= maxK; k++)
    {
        std::cout<<"Testing k "<<k<<std::endl;
        double start = elapsedTime();

        double accuracy = crossValidation.validate(k);
        std::cout<<"K: "<<k<<" Accuracy: "<<accuracy * 100<<"% completed in "<<elapsedTime() - start<<" seconds"<<std::endl;

        if(accuracy > bestAccuracy)
        {
            bestAccuracy = accuracy;
            bestK = k;
        }
    }

    return bestK;
}

// Igual que loadDataset de main.cc.
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    std::string img_path = "train/train-images-idx3-ubyte";
    std::string label_path = "train/train-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), train))
    {
        std::cout<<"The training data could not be read"<<std::endl;
        return false;
    }

    img_path = "test/t10k-images-idx3-ubyte";
    label_path = "test/t10k-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), test))
    {
        std::cout<<"The test data could not be read"<<std::endl;
        return false;
    }

    return true;
}

int main(int argc, char** argv)
{
    const unsigned minK = 1;
    unsigned maxK = 10, folds = 5;
    std::size_t tuningSamples = 20000;
    std::size_t testSamples = std::numeric_limits<std::size_t>::max();
    std::size_t finalTrainSamples = std::numeric_limits<std::size_t>::max();
    try
    {
        if (argc > 1) maxK = std::stoul(argv[1]);
        if (argc > 2) folds = std::stoul(argv[2]);
        if (argc > 3) tuningSamples = std::stoul(argv[3]);
        if (argc > 4) testSamples = std::stoul(argv[4]);
        if (argc > 5) finalTrainSamples = std::stoul(argv[5]);
    }
    catch (const std::exception &)
    {
        std::cerr << "Uso: knn_seq [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]" << std::endl;
        return 1;
    }
    if (maxK <= minK || folds < 2)
    {
        std::cerr << "Se necesita maxK > 1 y folds >= 2" << std::endl;
        return 1;
    }

    // Load MNIST dataset.
    double totalStart = elapsedTime();
    std::vector<TSample> train;
    std::vector<TSample> test;
    if(!loadDataset(train, test))
    {
        return 1;
    }
    const double tLoad = elapsedTime() - totalStart;

    const std::size_t nTuning = std::min(train.size(), tuningSamples);
    const std::size_t nTest = std::min(test.size(), testSamples);
    const std::size_t nFinalTrain = std::min(train.size(), finalTrainSamples);

    // Fine tune the KNN classifier with training data.
    double start = elapsedTime();
    int k = fineTuning(minK, maxK, std::vector<TSample>(train.begin(), train.begin() + nTuning), folds);
    const double tTune = elapsedTime() - start;
    std::cout<<"Best k: "<<k<<" found in "<<tTune<<" seconds"<<std::endl;

    // Test the classifier with testing data.
    start = elapsedTime();
    // Por defecto se usan los conjuntos completos (como main.cc); solo se copia un subconjunto si se pide uno menor.
    std::vector<TSample> reducedTrain, reducedTest;
    const std::vector<TSample> *finalTrain = &train;
    const std::vector<TSample> *finalTest = &test;
    if(nFinalTrain < train.size())
    {
        reducedTrain.assign(train.begin(), train.begin() + nFinalTrain);
        finalTrain = &reducedTrain;
    }
    if(nTest < test.size())
    {
        reducedTest.assign(test.begin(), test.begin() + nTest);
        finalTest = &reducedTest;
    }
    std::vector<int> labels;
    KNN knn(*finalTrain, k);
    double accuracy = knn.classifyAndEvaluate(*finalTest, labels);
    const double tFinal = elapsedTime() - start;

    std::cout<<"Test accuracy: "<<accuracy<<" completed in "<<tFinal<<" seconds"<<std::endl;

    const double total = elapsedTime() - totalStart;
    std::cout<<"Total elapsed time "<<total<<" seconds"<<std::endl;

    // Linea legible por maquina (la recoge run_all.sh). Columnas:
    // version,np,tuning,test,final_train,max_k,folds,best_k,test_acc,total_s,
    // load_s,tune_s,final_s,comm_s,nT,nE,restarts,wasted_s   (nT..wasted_s no aplican aqui)
    char csv[512];
    std::snprintf(csv, sizeof csv, "CSV,knn_seq" FLAVOR ",1,%zu,%zu,%zu,%u,%u,%d,%.6f,%.6f,%.6f,%.6f,%.6f,0.000000,,,,",
                  nTuning, nTest, nFinalTrain, maxK, folds, k, accuracy, total, tLoad, tTune, tFinal);
    std::cout<<csv<<std::endl;

    return 0;
}
#@@END knn_seq.cpp
#@@BEGIN knn_mpi.cpp
// =============================================================================
// knn_mpi.cpp  -  PASO 1: paralelismo de DATOS con MPI
//
// Versión paralela de main.cc (ajuste de K por validación cruzada + evaluación
// final con el mejor K). El código secuencial NO se modifica: esta versión
// reutiliza KNN, loadMnist y TSample tal cual.
//
// Estrategia
//   * Modelo: paso de mensajes, memoria distribuida. Cada proceso MPI tiene su
//     propia memoria, así que cada uno lee el dataset de disco (no hay que
//     enviarlo por la red).
//   * Ajuste de K: hay numK x folds = 10 x 5 = 50 tareas (k, fold)
//     independientes. Se reparten de forma CÍCLICA entre los procesos
//     (tarea t -> proceso t % size). Todas cuestan casi lo mismo.
//   * Los 50 resultados se juntan con UN SOLO MPI_Allreduce. Así todos los
//     procesos conocen todas las accuracies y eligen el mismo K sin más mensajes.
//   * Evaluación final: las muestras de test se reparten en bloques contiguos;
//     cada proceso cuenta sus aciertos y MPI_Reduce los suma en el proceso 0.
//
// Compilar:   mpic++ -O2 -std=c++17 -I. knn_mpi.cpp knn.cc mnist.cc sample.cc -o build/knn_mpi
// Ejecutar (desde la carpeta build/, donde están train/ y test/):
//             mpirun -np 4 ./knn_mpi
// Parámetros opcionales (para probar otros tamaños del problema):
//             mpirun -np 4 ./knn_mpi [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]
//             por defecto: 10 5 20000 todas todas (= problema completo)
//             Con el problema completo el ajuste pesa ~84 % y la evaluacion final ~16 %.
//             En pruebas pequenas, reduce tambien [entrenamiento_final] (p. ej. 4000 1000 12000)
//             para conservar una proporcion parecida.
// =============================================================================

#include <mpi.h>

#include "sample.h"
#include "knn.h"
#include "mnist.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef FLAVOR
#define FLAVOR "" // run_all.sh compila la variante con la biblioteca limpia con -DFLAVOR='"_opt"'
#endif

using namespace knn;
using namespace mnist;

namespace
{

// Aborta TODOS los procesos. Si solo terminara uno, los demás se quedarían
// bloqueados para siempre en la siguiente operación colectiva.
[[noreturn]] void abortAll(int rank, const std::string &msg)
{
    std::cerr << "[proceso " << rank << "] " << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

// Igual que loadDataset de main.cc: las rutas son relativas al directorio
// desde el que se lanza el programa (en todos los nodos).
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    return loadMnist("train/train-images-idx3-ubyte", "train/train-labels-idx1-ubyte", train)
        && loadMnist("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", test);
}

// Número de muestras de samples[begin, end) que el clasificador acierta.
long long countMatches(const KNN &knn, const std::vector<TSample> &samples,
                       std::size_t begin, std::size_t end)
{
    long long matches = 0;
    for (std::size_t i = begin; i < end; i++)
    {
        if (knn.classify(samples[i]) == samples[i].label)
            matches++;
    }
    return matches;
}

// Accuracy del KNN con 'k' vecinos usando el fold 'fold' como validación y el
// resto como entrenamiento. Misma partición que CrossValidation::splitDataset
// del código secuencial (mismos índices, mismo orden), así que los resultados
// son idénticos a los del secuencial.
double evaluateFold(const std::vector<TSample> &data, unsigned folds, unsigned fold, int k)
{
    const std::size_t start = data.size() * fold / folds;
    const std::size_t end = data.size() * (fold + 1) / folds;

    std::vector<TSample> train;
    train.reserve(data.size() - (end - start));
    train.insert(train.end(), data.begin(), data.begin() + start);
    train.insert(train.end(), data.begin() + end, data.end());

    KNN knn(train, k);
    return countMatches(knn, data, start, end) / (double)(end - start);
}

} // namespace

int main(int argc, char **argv)
{
    // ---- Arranque de MPI --------------------------------------------------
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // identificador de este proceso: 0 .. size-1
    MPI_Comm_size(MPI_COMM_WORLD, &size); // número total de procesos

    // ---- Parámetros del problema (por defecto, los de main.cc) -------------
    const unsigned minK = 1;
    unsigned maxK = 10, folds = 5;
    std::size_t tuningSamples = 20000;
    std::size_t testSamples = std::numeric_limits<std::size_t>::max();
    std::size_t finalTrainSamples = std::numeric_limits<std::size_t>::max();
    try
    {
        if (argc > 1) maxK = std::stoul(argv[1]);
        if (argc > 2) folds = std::stoul(argv[2]);
        if (argc > 3) tuningSamples = std::stoul(argv[3]);
        if (argc > 4) testSamples = std::stoul(argv[4]);
        if (argc > 5) finalTrainSamples = std::stoul(argv[5]);
    }
    catch (const std::exception &)
    {
        abortAll(rank, "Uso: knn_mpi [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]");
    }
    if (maxK <= minK || folds < 2)
        abortAll(rank, "Se necesita maxK > 1 y folds >= 2");

    // Todos los procesos empiezan a medir a la vez.
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    // ---- 1. Carga del dataset (la hace cada proceso, sin comunicación) ------
    std::vector<TSample> train, test;
    if (!loadDataset(train, test))
        abortAll(rank, "No se pudo leer MNIST: faltan las carpetas train/ y test/ en el directorio actual");
    const double tLoad = MPI_Wtime() - tStart;

    // Subconjunto de entrenamiento usado para el ajuste (igual que main.cc).
    const std::size_t nTuning = std::min(train.size(), tuningSamples);
    if (folds > nTuning)
        abortAll(rank, "Hay menos muestras de ajuste que folds");
    const std::vector<TSample> tuning(train.begin(), train.begin() + nTuning);

    // ---- 2. Ajuste de K: 50 tareas (k, fold) repartidas cíclicamente --------
    const int numK = maxK - minK + 1;
    const int numTasks = numK * folds;
    std::vector<double> acc(numTasks, 0.0); // acc[(k - minK) * folds + fold]
    int myTasks = 0;

    double t = MPI_Wtime();
    for (int task = rank; task < numTasks; task += size)
    {
        const int k = minK + task / folds;
        const int fold = task % folds;
        acc[task] = evaluateFold(tuning, folds, fold, k);
        myTasks++;
    }
    const double tTuneCompute = MPI_Wtime() - t;

    // Cada proceso solo ha rellenado SUS posiciones (el resto vale 0.0).
    // Sumando elemento a elemento obtenemos el vector completo y exacto
    // (x + 0.0 + 0.0 ... == x). Allreduce = Reduce + difusión del resultado.
    t = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, acc.data(), numTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double tTuneComm = MPI_Wtime() - t; // incluye esperar al proceso más lento

    // Todos los procesos eligen el K de la misma forma que el secuencial:
    // media de los folds en orden, y gana el mayor accuracy (en empate, el K menor).
    std::vector<double> accPerK(numK, 0.0);
    unsigned bestK = minK;
    double bestAccuracy = 0.0;
    for (int ki = 0; ki < numK; ki++)
    {
        double a = 0.0;
        for (unsigned f = 0; f < folds; f++)
            a += acc[ki * folds + f];
        a /= folds;
        accPerK[ki] = a;
        if (a > bestAccuracy)
        {
            bestAccuracy = a;
            bestK = minK + ki;
        }
    }

    // ---- 3. Evaluación final con el mejor K: bloques de muestras de test ----
    t = MPI_Wtime();
    // Entrenamiento de la evaluacion final: por defecto, todo el conjunto.
    const std::size_t nFinalTrain = std::min(train.size(), finalTrainSamples);
    std::vector<TSample> reducedTrain;
    const std::vector<TSample> *finalTrain = &train;
    if (nFinalTrain < train.size())
    {
        reducedTrain.assign(train.begin(), train.begin() + nFinalTrain);
        finalTrain = &reducedTrain;
    }
    const KNN knn(*finalTrain, bestK);
    const std::size_t nTest = std::min(test.size(), testSamples);
    const std::size_t begin = nTest * static_cast<std::size_t>(rank) / size;
    const std::size_t end = nTest * static_cast<std::size_t>(rank + 1) / size;
    const long long matches = countMatches(knn, test, begin, end);
    const double tFinalCompute = MPI_Wtime() - t;

    t = MPI_Wtime();
    long long totalMatches = 0;
    MPI_Reduce(&matches, &totalMatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    const double tFinalComm = MPI_Wtime() - t;

    // Fin: el tiempo total es el del proceso que termina el último.
    MPI_Barrier(MPI_COMM_WORLD);
    const double tTotal = MPI_Wtime() - tStart;

    // ---- 4. Recogida de tiempos en el proceso 0 ----------------------------
    // Para cada fase nos interesa el MÁXIMO entre procesos.
    double mine[6] = {tLoad, tTuneCompute, tTuneComm, tFinalCompute, tFinalComm, tTotal};
    double worst[6];
    MPI_Reduce(mine, worst, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Tiempo de cómputo del ajuste de cada proceso (para ver el equilibrio de carga).
    std::vector<double> tuneTimes(size);
    std::vector<int> tuneTasks(size);
    MPI_Gather(&tTuneCompute, 1, MPI_DOUBLE, tuneTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&myTasks, 1, MPI_INT, tuneTasks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Procesos MPI: " << size << " | K en [" << minK << ", " << maxK << "], " << folds
                  << " folds, " << nTuning << " muestras de ajuste, " << nTest << " de test, "
                  << nFinalTrain << " de entrenamiento final" << std::endl;
        // Mismo formato que el secuencial (6 cifras significativas) para poder compararlos.
        for (int ki = 0; ki < numK; ki++)
            std::cout << "K: " << minK + ki << " Accuracy: " << accPerK[ki] * 100 << "%" << std::endl;
        std::cout << "Best k: " << bestK << std::endl;
        std::cout << "Test accuracy: " << totalMatches / (double)nTest << std::endl;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Tiempos (s, maximo entre procesos):" << std::endl;
        std::cout << "  carga del dataset (replicada)   : " << worst[0] << std::endl;
        std::cout << "  ajuste de K - calculo           : " << worst[1] << std::endl;
        std::cout << "  ajuste de K - espera + Allreduce: " << worst[2] << std::endl;
        std::cout << "  evaluacion final - calculo      : " << worst[3] << std::endl;
        std::cout << "  evaluacion final - Reduce       : " << worst[4] << std::endl;
        std::cout << "Total elapsed time " << worst[5] << " seconds" << std::endl;

        std::cout << "Reparto del ajuste por proceso (tareas / segundos de calculo):" << std::endl;
        for (int r = 0; r < size; r++)
            std::cout << "  proceso " << r << ": " << tuneTasks[r] << " tareas, " << tuneTimes[r] << " s" << std::endl;

        // Linea legible por maquina (la recoge run_all.sh). Columnas:
        // version,np,tuning,test,final_train,max_k,folds,best_k,test_acc,total_s,
        // load_s,tune_s,final_s,comm_s,nT,nE,restarts,wasted_s   (nT..wasted_s no aplican aqui)
        char csv[512];
        std::snprintf(csv, sizeof csv, "CSV,knn_mpi" FLAVOR ",%d,%zu,%zu,%zu,%u,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,,,,",
                      size, nTuning, nTest, nFinalTrain, maxK, folds, bestK, totalMatches / (double)nTest,
                      worst[5], worst[0], worst[1], worst[3], worst[2] + worst[4]);
        std::cout << csv << std::endl;
    }

    MPI_Finalize();
    return 0;
}
#@@END knn_mpi.cpp
#@@BEGIN knn_mpi_spec.cpp
// =============================================================================
// knn_mpi_spec.cpp  -  PASOS 2 y 3: paralelismo FUNCIONAL + ejecucion ESPECULATIVA
//
// Secuencial:  ajuste de K (validacion cruzada)  ->  elegir K  ->  evaluacion final.
// La evaluacion final depende del K que salga del ajuste. Aqui se rompe esa
// dependencia "apostando": la evaluacion final EMPIEZA ya, con el mejor K
// conocido hasta ese momento, mientras el ajuste sigue ejecutandose.
//
// Dos grupos de procesos (MPI_Comm_split) que trabajan A LA VEZ:
//
//   GRUPO DE AJUSTE (rangos 0 .. nT-1)
//     Calcula las tareas (k, fold) y envia el resultado de cada una, segun la
//     acaba, al LIDER del grupo de evaluacion (MPI_Isend, tag TAG_RESULT).
//
//   GRUPO DE EVALUACION (rangos nT .. size-1; el lider es el rango nT)
//     Cada proceso clasifica su bloque de muestras de test con el mejor K
//     conocido (al principio, un K inicial supuesto). Cada pocas muestras mira
//     si ha llegado algun mensaje (MPI_Iprobe, no bloqueante):
//       - El lider recibe resultados del ajuste, detecta cuando se completa un K
//         (sus 5 folds) y, si es mejor que todos los anteriores, avisa al resto
//         del grupo con un mensaje (K, definitivo?) (tag TAG_BEST).
//       - Quien se entera de que su K ya no es el mejor DESCARTA lo hecho y
//         REINICIA con el K nuevo.
//     Al llegar el ultimo resultado, el lider envia el K definitivo. La
//     evaluacion termina cuando el bloque se ha clasificado con ese K.
//     Al final MPI_Reduce (en el comunicador del grupo) suma los aciertos.
//
// MODO (ultimo argumento)
//   0 = ESTATICO (paso 2): las tareas de ajuste se reparten de forma ciclica al
//       empezar y los procesos de evaluacion solo evaluan (y esperan).
//   1 = DINAMICO (paso 3): el lider tiene una cola de tareas de ajuste y las
//       reparte BAJO DEMANDA (TAG_REQ -> TAG_TASK). Asi el ajuste se equilibra
//       solo (no importa que P no divida a 50) y, ademas, los procesos de
//       evaluacion que terminan su bloque y esperan AYUDAN con tareas de ajuste
//       en vez de quedarse parados. Una tarea de ayuda se termina antes de
//       atender un cambio de K (para no desperdiciarla).
//
// La eleccion de K es DETERMINISTA (mayor accuracy; en empate, el K menor), asi
// que el resultado final es identico al secuencial aunque los resultados lleguen
// desordenados.
//
// Cuantos procesos van a cada grupo: por defecto, proporcional al trabajo de
// cada fase (con el problema completo, el ajuste ~84 % y la evaluacion ~16 %).
//
// Compilar:   mpic++ -O2 -std=c++17 -I. knn_mpi_spec.cpp knn.cc mnist.cc sample.cc -o build/knn_mpi_spec
// Ejecutar (desde build/):  mpirun -np 4 ./knn_mpi_spec     (minimo 2 procesos)
// Parametros opcionales:
//   mpirun -np 4 ./knn_mpi_spec [maxK] [folds] [muestras_ajuste] [muestras_test]
//                               [entrenamiento_final] [procesos_evaluacion] [k_inicial] [modo]
//   por defecto: 10 5 20000 todas todas 0(=automatico) 1 0
// =============================================================================

#include <mpi.h>

#include "sample.h"
#include "knn.h"
#include "mnist.h"

#include <unistd.h> // usleep

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef FLAVOR
#define FLAVOR "" // run_all.sh compila la variante con la biblioteca limpia con -DFLAVOR='"_opt"'
#endif

using namespace knn;
using namespace mnist;

namespace
{

const int TAG_RESULT = 1; // cualquiera -> lider: {numero de tarea, accuracy}
const int TAG_BEST = 2;   // lider -> resto de evaluacion: {K, es_definitivo}
const int TAG_REQ = 3;    // (dinamico) peticion de tarea: cualquiera -> lider
const int TAG_TASK = 4;   // (dinamico) respuesta del lider: {numero de tarea | -1 si no quedan}
const int CHUNK = 25;     // muestras que se clasifican entre dos consultas de mensajes

[[noreturn]] void abortAll(int rank, const std::string &msg)
{
    std::cerr << "[proceso " << rank << "] " << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    return loadMnist("train/train-images-idx3-ubyte", "train/train-labels-idx1-ubyte", train)
        && loadMnist("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", test);
}

long long countMatches(const KNN &knn, const std::vector<TSample> &samples,
                       std::size_t begin, std::size_t end)
{
    long long matches = 0;
    for (std::size_t i = begin; i < end; i++)
    {
        if (knn.classify(samples[i]) == samples[i].label)
            matches++;
    }
    return matches;
}

// Accuracy del KNN con 'k' vecinos usando el fold 'fold' como validacion. Misma
// particion que CrossValidation::splitDataset del codigo secuencial. Si se pasa
// 'poll', se llama cada CHUNK muestras (para atender mensajes mientras se trabaja).
double evaluateFold(const std::vector<TSample> &data, unsigned folds, unsigned fold, int k,
                    const std::function<void()> &poll = nullptr)
{
    const std::size_t start = data.size() * fold / folds;
    const std::size_t end = data.size() * (fold + 1) / folds;

    std::vector<TSample> train;
    train.reserve(data.size() - (end - start));
    train.insert(train.end(), data.begin(), data.begin() + start);
    train.insert(train.end(), data.begin() + end, data.end());

    KNN knn(train, k);
    long long matches = 0;
    for (std::size_t i = start; i < end;)
    {
        const std::size_t chunkEnd = std::min(i + static_cast<std::size_t>(CHUNK), end);
        matches += countMatches(knn, data, i, chunkEnd);
        i = chunkEnd;
        if (poll) poll();
    }
    return matches / (double)(end - start);
}

// Barrera que NO consume CPU mientras espera. Un MPI_Barrier normal "gira" al
// 100 % de CPU esperando; si hay mas procesos que nucleos, quitaria tiempo a
// los que aun estan trabajando y falsearia las medidas.
void sleepyBarrier(MPI_Comm comm)
{
    MPI_Request req;
    MPI_Ibarrier(comm, &req);
    int done = 0;
    while (!done)
    {
        MPI_Test(&req, &done, MPI_STATUS_IGNORE);
        if (!done) usleep(1000);
    }
}

// -----------------------------------------------------------------------------
// Lleva la cuenta de los resultados del ajuste que van llegando (solo el lider).
// -----------------------------------------------------------------------------
class BestKTracker
{
public:
    BestKTracker(int minK, int numK, unsigned folds)
        : minK_(minK), folds_(folds), acc_(numK * folds, 0.0), got_(numK, 0), accK_(numK, 0.0) {}

    // Registra el resultado de la tarea (k, fold). Devuelve true si, con ello,
    // cambia el mejor K conocido.
    bool add(int task, double accuracy)
    {
        acc_[task] = accuracy;
        received_++;
        const int ki = task / static_cast<int>(folds_);
        if (++got_[ki] < static_cast<int>(folds_))
            return false; // este K aun no tiene todos sus folds

        // Media de los folds EN ORDEN: mismos calculos (y mismos bits) que el secuencial.
        double a = 0.0;
        for (unsigned f = 0; f < folds_; f++)
            a += acc_[ki * folds_ + f];
        a /= folds_;
        accK_[ki] = a;

        if (bestKi_ < 0 || a > accK_[bestKi_] || (a == accK_[bestKi_] && ki < bestKi_))
        {
            bestKi_ = ki;
            return true;
        }
        return false;
    }

    bool complete() const { return received_ == static_cast<int>(acc_.size()); }
    int bestK() const { return minK_ + bestKi_; }
    double accuracyOf(int ki) const { return accK_[ki]; }

private:
    int minK_;
    unsigned folds_;
    std::vector<double> acc_; // acc_[(k - minK) * folds + fold]
    std::vector<int> got_;    // folds recibidos de cada K
    std::vector<double> accK_;
    int received_ = 0;
    int bestKi_ = -1;
};

struct TimelineEntry
{
    double t;   // segundos desde el inicio
    int k;      // mejor K conocido en ese momento
    bool final; // true si el ajuste ya habia terminado
};

struct EvalStats
{
    int restarts = 0;       // veces que se descarto el trabajo por llegar un K distinto
    double compute = 0.0;   // segundos clasificando test (incluye el trabajo descartado)
    double wasted = 0.0;    // segundos de trabajo descartado
    double wait = 0.0;      // segundos esperando a que termine el ajuste
    int helpTasks = 0;      // (dinamico) tareas de ajuste hechas por este proceso de evaluacion
    double helpCompute = 0.0;
    std::vector<TimelineEntry> timeline; // solo el lider
};

// -----------------------------------------------------------------------------
// GRUPO DE AJUSTE, modo estatico: tareas repartidas ciclicamente de antemano.
// Devuelve los segundos de calculo y, en 'done', cuantas tareas ha hecho.
// -----------------------------------------------------------------------------
double runTuningStatic(const std::vector<TSample> &tuning, unsigned folds, int minK, int numTasks,
                       int myIndex, int nT, int leaderRank, int &done)
{
    std::vector<int> tasks;
    for (int task = myIndex; task < numTasks; task += nT) // reparto ciclico
        tasks.push_back(task);

    std::vector<std::array<double, 2>> msgs(tasks.size()); // buffers que deben seguir vivos hasta el Waitall
    std::vector<MPI_Request> reqs(tasks.size());
    double compute = 0.0;

    for (std::size_t idx = 0; idx < tasks.size(); idx++)
    {
        const int task = tasks[idx];
        const double t0 = MPI_Wtime();
        const double acc = evaluateFold(tuning, folds, task % folds, minK + task / folds);
        compute += MPI_Wtime() - t0;

        msgs[idx] = {static_cast<double>(task), acc};
        // Envio no bloqueante: sigo con la siguiente tarea sin esperar al lider.
        MPI_Isend(msgs[idx].data(), 2, MPI_DOUBLE, leaderRank, TAG_RESULT, MPI_COMM_WORLD, &reqs[idx]);
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    done = static_cast<int>(tasks.size());
    return compute;
}

// -----------------------------------------------------------------------------
// GRUPO DE AJUSTE, modo dinamico: pide tareas al lider hasta que le contesta -1.
// -----------------------------------------------------------------------------
double runTuningDynamic(const std::vector<TSample> &tuning, unsigned folds, int minK,
                        int leaderRank, int &done)
{
    std::deque<std::array<double, 2>> msgs;
    std::vector<MPI_Request> reqs;
    double compute = 0.0;
    done = 0;

    for (;;)
    {
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, leaderRank, TAG_REQ, MPI_COMM_WORLD);

        // Espera la respuesta durmiendo (el lider tarda a lo sumo unas decimas en atender).
        int task = -1, flag = 0;
        for (;;)
        {
            MPI_Iprobe(leaderRank, TAG_TASK, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if (flag) break;
            usleep(1000);
        }
        MPI_Recv(&task, 1, MPI_INT, leaderRank, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (task < 0)
            break; // no quedan tareas

        const double t0 = MPI_Wtime();
        const double acc = evaluateFold(tuning, folds, task % folds, minK + task / folds);
        compute += MPI_Wtime() - t0;
        done++;

        msgs.push_back({static_cast<double>(task), acc});
        reqs.emplace_back();
        MPI_Isend(msgs.back().data(), 2, MPI_DOUBLE, leaderRank, TAG_RESULT, MPI_COMM_WORLD, &reqs.back());
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    return compute;
}

// -----------------------------------------------------------------------------
// GRUPO DE EVALUACION: clasifica begin..end con el mejor K conocido y reinicia
// si cambia. En modo dinamico, ayuda con el ajuste cuando se queda sin trabajo.
// -----------------------------------------------------------------------------
class Evaluator
{
public:
    struct Config
    {
        const std::vector<TSample> *train = nullptr;  // entrenamiento de la evaluacion final
        const std::vector<TSample> *test = nullptr;
        const std::vector<TSample> *tuning = nullptr; // datos del ajuste (solo para ayudar)
        std::size_t begin = 0, end = 0;               // bloque de test de este proceso
        int k0 = 1;                                   // K inicial supuesto
        int leaderRank = 0;
        bool isLeader = false;
        std::vector<int> workers;                     // resto del grupo (solo lo usa el lider)
        BestKTracker *tracker = nullptr;              // solo el lider
        double tStart = 0.0;
        bool dynamic = false;
        unsigned folds = 5;
        int minK = 1;
        int numTasks = 0;
    };

    Evaluator(const Config &c, EvalStats &stats) : c_(c), stats_(stats), kWanted_(c.k0) {}

    // Devuelve los aciertos del bloque con el K definitivo (en finalK).
    long long run(int &finalK)
    {
        int kUsed = -1;          // K con el que se esta clasificando de verdad
        long long matches = 0;
        std::size_t i = c_.begin;
        double workThisK = 0.0;  // trabajo hecho con kUsed (se pierde si hay que reiniciar)
        std::unique_ptr<KNN> knn;
        bool canHelp = c_.dynamic;

        for (;;)
        {
            if (kUsed != kWanted_) // (re)inicio: o es la primera vez, o ha llegado un K distinto
            {
                if (kUsed != -1)
                {
                    stats_.restarts++;
                    stats_.wasted += workThisK; // lo hecho con el K anterior se descarta
                }
                const double t0 = MPI_Wtime();
                kUsed = kWanted_;
                knn.reset(new KNN(*c_.train, kUsed));
                matches = 0;
                i = c_.begin;
                workThisK = MPI_Wtime() - t0;
                stats_.compute += workThisK;
            }

            if (i < c_.end) // quedan muestras: un trozo de CHUNK y luego miro mensajes
            {
                const double t0 = MPI_Wtime();
                const std::size_t chunkEnd = std::min(i + static_cast<std::size_t>(CHUNK), c_.end);
                matches += countMatches(*knn, *c_.test, i, chunkEnd);
                i = chunkEnd;
                const double dt = MPI_Wtime() - t0;
                workThisK += dt;
                stats_.compute += dt;
                poll();
                continue;
            }

            // Bloque terminado con kUsed.
            if (finalKnown_ && kWanted_ == kUsed)
                break; // el K usado es el definitivo: hecho

            // Aun puede cambiar el K. Modo dinamico: en vez de esperar, ayudo con el ajuste.
            if (canHelp)
            {
                const int task = fetchTask();
                if (task >= 0)
                {
                    helpWith(task);
                    continue; // puede haber cambiado el K mientras ayudaba
                }
                canHelp = false; // no quedan tareas
            }

            // Espero mensajes sin ocupar CPU.
            const double t0 = MPI_Wtime();
            while (kWanted_ == kUsed && !finalKnown_)
            {
                if (!poll())
                    usleep(1000);
            }
            stats_.wait += MPI_Wtime() - t0;
        }

        MPI_Waitall(static_cast<int>(outReq_.size()), outReq_.data(), MPI_STATUSES_IGNORE);
        finalK = kUsed;
        return matches;
    }

    // Barrera final. El lider sigue atendiendo peticiones de tarea mientras espera:
    // asi nadie se queda esperando una respuesta que no llega.
    void finalBarrier(bool serve)
    {
        if (!serve)
        {
            sleepyBarrier(MPI_COMM_WORLD);
            return;
        }
        MPI_Request req;
        MPI_Ibarrier(MPI_COMM_WORLD, &req);
        int done = 0;
        while (!done)
        {
            MPI_Test(&req, &done, MPI_STATUS_IGNORE);
            if (!done && !poll())
                usleep(1000);
        }
    }

private:
    // Envia el aviso (K, definitivo?) a todo el grupo de evaluacion (solo el lider).
    void announce(int k, bool fin)
    {
        for (int w : c_.workers)
        {
            outBuf_.push_back({k, fin ? 1 : 0});
            outReq_.emplace_back();
            MPI_Isend(outBuf_.back().data(), 2, MPI_INT, w, TAG_BEST, MPI_COMM_WORLD, &outReq_.back());
        }
    }

    // El lider registra un resultado de ajuste (llegado por mensaje o calculado por el).
    void onResult(int task, double accuracy)
    {
        const bool changed = c_.tracker->add(task, accuracy);
        if (c_.tracker->complete())
        {
            kWanted_ = c_.tracker->bestK();
            finalKnown_ = true;
            stats_.timeline.push_back({MPI_Wtime() - c_.tStart, kWanted_, true});
            announce(kWanted_, true);
        }
        else if (changed)
        {
            kWanted_ = c_.tracker->bestK();
            stats_.timeline.push_back({MPI_Wtime() - c_.tStart, kWanted_, false});
            announce(kWanted_, false);
        }
    }

    // Consulta mensajes sin bloquear. Devuelve true si ha procesado alguno.
    bool poll()
    {
        bool got = false;
        int flag = 0;
        if (c_.isLeader)
        {
            for (;;) // resultados del ajuste
            {
                MPI_Status st;
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &st);
                if (!flag) break;
                double msg[2];
                MPI_Recv(msg, 2, MPI_DOUBLE, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                got = true;
                onResult(static_cast<int>(msg[0]), msg[1]);
            }
            if (c_.dynamic)
            {
                for (;;) // peticiones de tarea: SIEMPRE se contestan
                {
                    MPI_Status st;
                    MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQ, MPI_COMM_WORLD, &flag, &st);
                    if (!flag) break;
                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    got = true;
                    int task = nextTask_ < c_.numTasks ? nextTask_++ : -1;
                    MPI_Send(&task, 1, MPI_INT, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            for (;;) // avisos del lider
            {
                MPI_Iprobe(c_.leaderRank, TAG_BEST, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
                if (!flag) break;
                int msg[2];
                MPI_Recv(msg, 2, MPI_INT, c_.leaderRank, TAG_BEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                got = true;
                kWanted_ = msg[0];
                finalKnown_ = (msg[1] != 0);
            }
        }
        return got;
    }

    // Modo dinamico: obtiene una tarea de ajuste (-1 si no quedan).
    int fetchTask()
    {
        if (c_.isLeader)
            return nextTask_ < c_.numTasks ? nextTask_++ : -1;

        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, c_.leaderRank, TAG_REQ, MPI_COMM_WORLD);
        for (;;) // la respuesta SIEMPRE llega; mientras tanto sigo atendiendo avisos de K
        {
            int flag = 0;
            MPI_Iprobe(c_.leaderRank, TAG_TASK, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if (flag)
            {
                int task;
                MPI_Recv(&task, 1, MPI_INT, c_.leaderRank, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                return task;
            }
            if (!poll())
                usleep(1000);
        }
    }

    // Calcula una tarea de ajuste y entrega el resultado al lider (atendiendo mensajes mientras tanto).
    void helpWith(int task)
    {
        const double t0 = MPI_Wtime();
        const double acc = evaluateFold(*c_.tuning, c_.folds, task % c_.folds, c_.minK + task / c_.folds,
                                        [this]() { poll(); });
        stats_.helpCompute += MPI_Wtime() - t0;
        stats_.helpTasks++;

        if (c_.isLeader)
        {
            onResult(task, acc);
        }
        else
        {
            outRes_.push_back({static_cast<double>(task), acc});
            outReq_.emplace_back();
            MPI_Isend(outRes_.back().data(), 2, MPI_DOUBLE, c_.leaderRank, TAG_RESULT, MPI_COMM_WORLD,
                      &outReq_.back());
        }
    }

    Config c_;
    EvalStats &stats_;
    int kWanted_;            // el K que ahora mismo se cree el mejor
    bool finalKnown_ = false; // true cuando el ajuste ha terminado y kWanted_ es el definitivo
    int nextTask_ = 0;       // (lider, dinamico) siguiente tarea de ajuste por repartir

    std::deque<std::array<int, 2>> outBuf_;      // buffers de envios: deben seguir vivos hasta el Waitall
    std::deque<std::array<double, 2>> outRes_;
    std::vector<MPI_Request> outReq_;
};

const int NFIELDS = 9; // campos de metricas por proceso (ver main)

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2)
        abortAll(rank, "Esta version necesita al menos 2 procesos (un grupo de ajuste y otro de evaluacion)");

    // ---- Parametros ---------------------------------------------------------
    const unsigned minK = 1;
    unsigned maxK = 10, folds = 5;
    std::size_t tuningSamples = 20000;
    std::size_t testSamples = std::numeric_limits<std::size_t>::max();
    std::size_t finalTrainSamples = std::numeric_limits<std::size_t>::max();
    int evalProcs = 0;       // 0 = automatico
    unsigned initialK = minK;
    unsigned mode = 0;       // 0 = estatico (paso 2), 1 = dinamico (paso 3)
    try
    {
        if (argc > 1) maxK = std::stoul(argv[1]);
        if (argc > 2) folds = std::stoul(argv[2]);
        if (argc > 3) tuningSamples = std::stoul(argv[3]);
        if (argc > 4) testSamples = std::stoul(argv[4]);
        if (argc > 5) finalTrainSamples = std::stoul(argv[5]);
        if (argc > 6) evalProcs = std::stoi(argv[6]);
        if (argc > 7) initialK = std::stoul(argv[7]);
        if (argc > 8) mode = std::stoul(argv[8]);
    }
    catch (const std::exception &)
    {
        abortAll(rank, "Uso: knn_mpi_spec [maxK] [folds] [muestras_ajuste] [muestras_test] "
                       "[entrenamiento_final] [procesos_evaluacion] [k_inicial] [modo 0|1]");
    }
    if (maxK <= minK || folds < 2)
        abortAll(rank, "Se necesita maxK > 1 y folds >= 2");
    if (initialK < minK || initialK > maxK)
        abortAll(rank, "k_inicial debe estar entre 1 y maxK");
    if (mode > 1)
        abortAll(rank, "modo debe ser 0 (estatico) o 1 (dinamico)");
    const bool dynamic = (mode == 1);

    sleepyBarrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    // ---- Carga del dataset (cada proceso, sin comunicacion) ------------------
    std::vector<TSample> train, test;
    if (!loadDataset(train, test))
        abortAll(rank, "No se pudo leer MNIST: faltan las carpetas train/ y test/ en el directorio actual");
    const double tLoad = MPI_Wtime() - tStart;

    const std::size_t nTuning = std::min(train.size(), tuningSamples);
    if (folds > nTuning)
        abortAll(rank, "Hay menos muestras de ajuste que folds");
    const std::vector<TSample> tuning(train.begin(), train.begin() + nTuning);

    const std::size_t nTest = std::min(test.size(), testSamples);
    const std::size_t nFinalTrain = std::min(train.size(), finalTrainSamples);
    std::vector<TSample> reducedTrain;
    const std::vector<TSample> *finalTrain = &train;
    if (nFinalTrain < train.size())
    {
        reducedTrain.assign(train.begin(), train.begin() + nFinalTrain);
        finalTrain = &reducedTrain;
    }

    // ---- Reparto en dos grupos ----------------------------------------------
    const int numK = maxK - minK + 1;
    const int numTasks = numK * folds;

    // Trabajo de cada fase en "distancias calculadas": ajuste = numK * n^2 * (folds-1)/folds;
    // evaluacion final = test * entrenamiento_final.
    const double tuningWork = double(numK) * nTuning * nTuning * (folds - 1) / folds;
    const double finalWork = double(nTest) * nFinalTrain;
    int nE = evalProcs > 0 ? evalProcs : static_cast<int>(std::lround(size * finalWork / (finalWork + tuningWork)));
    nE = std::max(1, std::min(nE, size - 1));
    const int nT = size - nE;

    const bool isEval = rank >= nT;
    const int leaderRank = nT; // primer proceso del grupo de evaluacion
    const bool isLeader = (rank == leaderRank);

    // Un comunicador por grupo: las colectivas del grupo de evaluacion solo lo implican a el.
    MPI_Comm groupComm;
    MPI_Comm_split(MPI_COMM_WORLD, isEval ? 1 : 0, rank, &groupComm);

    // ---- Cada proceso hace lo suyo -------------------------------------------
    EvalStats stats;
    double tuneCompute = 0.0; // segundos en tareas de ajuste (grupo de ajuste)
    int tuneTasks = 0;
    long long matches = 0;
    int finalK = -1;
    std::unique_ptr<BestKTracker> tracker;
    std::unique_ptr<Evaluator> evaluator;

    if (!isEval)
    {
        tuneCompute = dynamic ? runTuningDynamic(tuning, folds, minK, leaderRank, tuneTasks)
                              : runTuningStatic(tuning, folds, minK, numTasks, rank, nT, leaderRank, tuneTasks);
    }
    else
    {
        const int er = rank - nT; // indice dentro del grupo de evaluacion

        Evaluator::Config cfg;
        cfg.train = finalTrain;
        cfg.test = &test;
        cfg.tuning = &tuning;
        cfg.begin = nTest * static_cast<std::size_t>(er) / nE;
        cfg.end = nTest * static_cast<std::size_t>(er + 1) / nE;
        cfg.k0 = initialK;
        cfg.leaderRank = leaderRank;
        cfg.isLeader = isLeader;
        cfg.tStart = tStart;
        cfg.dynamic = dynamic;
        cfg.folds = folds;
        cfg.minK = minK;
        cfg.numTasks = numTasks;
        if (isLeader)
        {
            tracker.reset(new BestKTracker(minK, numK, folds));
            cfg.tracker = tracker.get();
            for (int r = nT + 1; r < size; r++)
                cfg.workers.push_back(r);
        }

        evaluator.reset(new Evaluator(cfg, stats));
        matches = evaluator->run(finalK);
        tuneCompute = stats.helpCompute;
        tuneTasks = stats.helpTasks;
    }
    const double tDone = MPI_Wtime() - tStart; // cuando termina ESTE proceso su trabajo

    // Barrera final. Los procesos de evaluacion siguen atendiendo peticiones hasta que todos llegan.
    if (isEval)
        evaluator->finalBarrier(isLeader);
    else
        sleepyBarrier(MPI_COMM_WORLD);

    long long totalMatches = 0;
    if (isEval)
        MPI_Reduce(&matches, &totalMatches, 1, MPI_LONG_LONG, MPI_SUM, 0, groupComm); // 0 = lider

    // ---- Recogida de metricas en el lider -----------------------------------
    // {es_evaluacion, fin, calculo_evaluacion, espera, descartado, reinicios, carga, tareas_ajuste, calculo_ajuste}
    const double mine[NFIELDS] = {isEval ? 1.0 : 0.0, tDone, isEval ? stats.compute : 0.0, stats.wait,
                                  stats.wasted, static_cast<double>(stats.restarts), tLoad,
                                  static_cast<double>(tuneTasks), tuneCompute};
    std::vector<double> all(NFIELDS * size);
    MPI_Gather(mine, NFIELDS, MPI_DOUBLE, all.data(), NFIELDS, MPI_DOUBLE, leaderRank, MPI_COMM_WORLD);

    if (isLeader)
    {
        double total = 0.0, maxLoad = 0.0, wastedSum = 0.0;
        for (int r = 0; r < size; r++)
        {
            const double *v = &all[NFIELDS * r];
            total = std::max(total, v[1]);
            maxLoad = std::max(maxLoad, v[6]);
            if (v[0] > 0.5) wastedSum += v[4];
        }

        std::cout << "Procesos MPI: " << size << " (ajuste: " << nT << ", evaluacion: " << nE << ") | modo "
                  << (dynamic ? "dinamico" : "estatico") << " | K en [" << minK << ", " << maxK << "], " << folds
                  << " folds, " << nTuning << " muestras de ajuste, " << nTest << " de test, " << nFinalTrain
                  << " de entrenamiento final, K inicial " << initialK << std::endl;
        // Mismo formato que el secuencial y el paso 1 (6 cifras significativas).
        for (int ki = 0; ki < numK; ki++)
            std::cout << "K: " << minK + ki << " Accuracy: " << tracker->accuracyOf(ki) * 100 << "%" << std::endl;
        std::cout << "Best k: " << finalK << std::endl;
        std::cout << "Test accuracy: " << totalMatches / (double)nTest << std::endl;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Evolucion del mejor K conocido (segundos desde el inicio):" << std::endl;
        std::cout << "  t=" << 0.0 << "  K inicial supuesto = " << initialK << std::endl;
        for (const TimelineEntry &e : stats.timeline)
            std::cout << "  t=" << e.t << "  " << (e.final ? "ajuste completo, K definitivo = " : "nuevo mejor K = ")
                      << e.k << std::endl;
        std::cout << "Reinicios de la evaluacion (lider): " << stats.restarts
                  << ", trabajo descartado (suma de evaluadores): " << wastedSum << " s" << std::endl;

        std::cout << "Por proceso (segundos):" << std::endl;
        for (int r = 0; r < size; r++)
        {
            const double *v = &all[NFIELDS * r];
            std::cout << "  proceso " << r << (v[0] > 0.5 ? " [evaluacion]" : " [ajuste]    ")
                      << "  termina a t=" << v[1];
            if (v[0] > 0.5)
            {
                std::cout << "  evaluacion=" << v[2] << " (descartado=" << v[4] << ")  espera=" << v[3]
                          << "  reinicios=" << static_cast<int>(v[5]);
                if (dynamic)
                    std::cout << "  ayuda al ajuste: " << static_cast<int>(v[7]) << " tareas, " << v[8] << " s";
            }
            else
            {
                std::cout << "  tareas=" << static_cast<int>(v[7]) << "  calculo=" << v[8];
            }
            std::cout << std::endl;
        }
        std::cout << "Carga del dataset (replicada, maximo): " << maxLoad << std::endl;
        std::cout << "Total elapsed time " << total << " seconds" << std::endl;

        // Linea legible por maquina (la recoge run_all.sh). Mismas columnas que knn_mpi;
        // aqui no hay tune_s / final_s / comm_s (las fases se solapan) pero si grupos y reinicios.
        const std::string version = std::string(dynamic ? "knn_mpi_dyn" : "knn_mpi_spec") + FLAVOR;
        char csv[512];
        std::snprintf(csv, sizeof csv, "CSV,%s,%d,%zu,%zu,%zu,%u,%u,%d,%.6f,%.6f,%.6f,,,,%d,%d,%d,%.6f",
                      version.c_str(), size, nTuning, nTest, nFinalTrain, maxK, folds, finalK,
                      totalMatches / (double)nTest, total, maxLoad, nT, nE, stats.restarts, wastedSum);
        std::cout << csv << std::endl;
    }

    MPI_Comm_free(&groupComm);
    MPI_Finalize();
    return 0;
}
#@@END knn_mpi_spec.cpp
#@@BEGIN graficas.py
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
#@@END graficas.py
#@@BEGIN opt/knn.h
#ifndef __KNN_H
#define __KNN_H

// VERSION LIMPIA (Tarea 0) de knn.h. Mismo interfaz y mismos resultados que el original.
// Cambio: el constructor copia el vector de entrenamiento de una sola vez (reserva y copia)
// en lugar de hacer push_back muestra a muestra, que obliga al vector a crecer y recolocarse.

#include "sample.h"

#include <vector>

namespace knn
{
    /// <summary>
    /// K-Nearest Neighbors classifier implementation.
    /// </summary>
    class KNN
    {
        public:
            KNN(const std::vector<TSample> &train, int k=1) : train(train), k(k) {}

            int classify(const TSample &test) const;

            double classifyAndEvaluate(const std::vector<TSample> &test, std::vector<int> &labels) const;

        private:
            /// <summary>
            /// The training data samples.
            /// </summary>
            std::vector<TSample> train;

            /// <summary>
            /// The k parameter.
            /// </summary>
            int k;
    };
}

#endif
#@@END opt/knn.h
#@@BEGIN opt/knn.cc
// VERSION LIMPIA (Tarea 0) de knn.cc. Mismos resultados que el original, bit a bit:
// no se cambia ni el calculo de distancias, ni el orden de los candidatos, ni el std::sort
// (otro algoritmo de seleccion podria romper los empates de forma distinta).
// Cambios, todos de "gastos generales" que no se pueden paralelizar ni aportan nada:
//  1. classify(): el vector de resultados se reserva una vez (y se reutiliza entre llamadas)
//     en vez de crecer con push_back en cada clasificacion (reasignaciones y copias).
//  2. maxVotesLabel(): recibe el mapa por referencia constante (antes lo copiaba entero).
//  3. classifyAndEvaluate(): se elimina el cout + flush por CADA muestra (E/S en el bucle
//     caliente; con varios procesos MPI ademas mezclaria las salidas) y se reserva 'labels'.

#include "knn.h"
#include "mnist.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace knn;

float euclidean(const TSample &a, const TSample &b)
{
    if (a.image.rows != b.image.rows || a.image.cols != b.image.cols)
        throw std::invalid_argument("Image dimensions differ");
    float sum = 0.0f;
    const float *imageA = a.image.pixels.data();
    const float *imageB = b.image.pixels.data();

    for(int i=0;i<a.image.rows*a.image.cols;i++)
    {
    	float diff = imageA[i] - imageB[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

typedef struct KNNResult
{
    KNNResult()
    {
        this->label = 0;
        this->distance = 0.0;
    };

    KNNResult(int label, float distance)
    {
        this->label = label;
        this->distance = distance;
    };

    KNNResult(const TSample &testSample, const TSample &trainingSample)
    {
        this->label = trainingSample.label;
        this->distance = euclidean(testSample, trainingSample);
    };

    bool operator<(const KNNResult &b) const
    {
        return this->distance < b.distance;
    }

    int label;
    float distance;
}KNNResult;

std::map<int, int> histogram(const std::vector<KNNResult> &results, int k)
{
    std::map<int, int> votes;
    for(int i=0;i<k;i++)
    {
        votes[results[i].label]++;
    }

    return votes;
}

// (limpieza 2) por referencia constante
int maxVotesLabel(const std::map<int, int> &histogram)
{
    int maxVotes = 0;
    int maxLabel = 0;
    std::map<int, int>::const_iterator itr;
    for (itr = histogram.begin(); itr != histogram.end(); ++itr) {
        if(itr->second > maxVotes) {
            maxVotes = itr->second;
            maxLabel = itr->first;
        }
    }

    return maxLabel;
}

int KNN::classify(const TSample &test) const
{
    if (k < 1 || static_cast<std::size_t>(k) > train.size())
        throw std::invalid_argument("k must be between 1 and the training size");

    // (limpieza 1) un unico buffer por proceso, reservado una vez y reutilizado
    static thread_local std::vector<KNNResult> results;
    results.clear();
    results.reserve(this->train.size());
    for(std::size_t i=0;i<this->train.size();i++)
    {
        results.emplace_back(test, this->train[i]);
    }

    std::sort(results.begin(), results.end());

    return maxVotesLabel(histogram(results, this->k));
}

double KNN::classifyAndEvaluate(const std::vector<TSample> &test, std::vector<int> &labels) const
{
    if (test.empty()) throw std::invalid_argument("Test set is empty");
    int matches = 0;
    labels.reserve(labels.size() + test.size()); // (limpieza 3)
    for(std::size_t i=0;i<test.size();i++)
    {
        // (limpieza 3) sin cout << i << "/" << ... << std::flush por muestra
        int label = this->classify(test[i]);
        labels.push_back(label);

        if(label == test[i].label) matches++;
    }

    return matches / (double)test.size();
}
#@@END opt/knn.cc
#@@BEGIN opt/crossValidation.h
#ifndef __CROSS_VALIDATION_H__
#define __CROSS_VALIDATION_H__

// VERSION LIMPIA (Tarea 0) de crossValidation.h. Mismo interfaz y resultados que el original.
// Cambio: el constructor copia el dataset de una vez (en vez de push_back uno a uno).

#include "sample.h"

#include <vector>
#include <stdexcept>
#include <string>

namespace crossValidation
{
    class CrossValidation
    {
        public:
            CrossValidation(const std::vector<TSample> &dataset, unsigned int folds)
            {
                if (folds < 2 || folds > dataset.size())
                    throw std::invalid_argument("folds must be between 2 and dataset size");
                this->dataset = dataset;
                this->folds = folds;
            };

            double validate(unsigned int k) const;

            static CrossValidation NFold(const std::vector<TSample> &dataset, unsigned int folds)
            {
                return CrossValidation(dataset, folds);
            };

            static CrossValidation LeaveOneOut(const std::vector<TSample> &dataset)
            {
                return CrossValidation(dataset, dataset.size());
            };

            std::string getName() const;

        private:
            std::vector<TSample> dataset;
            unsigned int folds;

            void splitDataset(std::vector<TSample> &train, std::vector<TSample> &validation, unsigned int fold) const;
    };
}

#endif
#@@END opt/crossValidation.h
#@@BEGIN opt/crossValidation.cc
// VERSION LIMPIA (Tarea 0) de crossValidation.cc. Mismos resultados y mismas salidas que el original.
// Cambio: splitDataset() reserva el tamano final de 'train' y 'validation' (antes crecian con
// push_back, recolocando y copiando miles de muestras cada vez que se llama; se llama una vez
// por cada par (k, fold)).

#include "crossValidation.h"
#include "sample.h"
#include "knn.h"

#include "timer.h"

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <string>
#include <sstream>

using namespace knn;
using namespace crossValidation;

double CrossValidation::validate(unsigned int k) const
{
    double accuracy = 0.0;
    std::cout<<"Starting "<<this->getName()<<std::endl;
    for(int fold=0;fold<this->folds;fold++)
    {
        std::vector<TSample> train, test;
        this->splitDataset(train, test, fold);

        std::cout<<"Testing fold "<<fold<<std::endl;
        double start = elapsedTime();
        KNN knn(train, k);
        std::vector<int> labels;
        double acc = knn.classifyAndEvaluate(test, labels);
        accuracy += acc;
        std::cout<<"Fold: "<<fold<<" Accuracy: "<<acc * 100<<"% completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
    }

    return accuracy /= this->folds;
}

void CrossValidation::splitDataset(std::vector<TSample> &train, std::vector<TSample> &validation, unsigned int fold) const
{
    assert(fold < this->folds && "Invalid fold");

    int start = this->dataset.size() * fold / this->folds;
    int end = this->dataset.size() * (fold + 1) / this->folds;

    train.reserve(this->dataset.size() - (end - start));
    validation.reserve(end - start);
    for(int i=0;i<this->dataset.size();i++)
    {
        if(i<start || i>=end)
            train.push_back(this->dataset[i]);
        else
            validation.push_back(this->dataset[i]);
    }
}

std::string CrossValidation::getName() const
{
    std::stringstream ss("");
    if(this->folds < this->dataset.size())
        ss<<folds<<"-Folds Cross Validation";
    else
        ss<<"leave-one-out Cross Validation";

    return ss.str();
}
#@@END opt/crossValidation.cc
#@@BEGIN tarea0/tarea0_vector.cpp
// Tarea 0 (1/2): vector dinamico con push_back (A) frente a vector con tamano fijo (B).
// Misma logica que el codigo del enunciado. Cambios minimos: SIZE se puede fijar con -DSIZE=...
// y se imprime una suma de comprobacion para que el compilador (-O2) no elimine los bucles.
// Compilar: g++ -O0|-O2 -DSIZE=1000000 tarea0_vector.cpp -o tarea0_vector
#include <iostream>
#include <vector>
#include <ctime>

#ifndef SIZE
#define SIZE 1000000
#endif

using namespace std;

int main()
{
    // A:
    clock_t begin = clock();
    // declaramos un vector sin inicializar.
    std::vector<int> a;
    // usamos push_back para anadir cada elemento.
    for(int i=0;i<SIZE;i++)
    {
        a.push_back(i);
    }
    double aTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<aTime<<"s"<<endl;

    // B:
    begin = clock();
    // declaramos un vector inicializado al tamano requerido.
    std::vector<int> b(SIZE);
    for(int i=0;i<SIZE;i++)
    {
        b[i] = i;
    }
    double bTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<bTime<<"s"<<endl;
    cout<<"Gain: "<<aTime/bTime<<endl;
    cout<<"Check: "<<(a[SIZE-1] + b[SIZE-1])<<endl;
}
#@@END tarea0/tarea0_vector.cpp
#@@BEGIN tarea0/tarea0_matriz.cpp
// Tarea 0 (2/2): recorrido de una matriz por columnas (A) frente a por filas (B).
// Misma logica que el codigo del enunciado. Cambios minimos: SIZE se fija con -DSIZE=...; las
// matrices son estaticas (en el enunciado estan en la pila y con SIZE grande la desbordan) y se
// imprime una suma de comprobacion para que el compilador (-O2) no elimine los bucles.
// Compilar: g++ -O0|-O2 -DSIZE=2000 tarea0_matriz.cpp -o tarea0_matriz
#include <iostream>
#include <vector>
#include <ctime>

#ifndef SIZE
#define SIZE 1000
#endif

using namespace std;

static int a[SIZE][SIZE];
static int b[SIZE][SIZE];

int main()
{
    // A: una matriz en C se almacena como un array de una unica dimension por filas.
    // Si la recorremos por columnas visitamos las celdas en el orden 1 4 7 2 5 8 3 6 9 ...
    clock_t begin = clock();
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
            a[j][i] = i+j;
    }
    double aTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<aTime<<"s"<<endl;

    // B: por filas visitamos las celdas en el orden 1 2 3 4 5 6 7 8 9 ...
    begin = clock();
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
            b[i][j] = i+j;
    }
    double bTime = double(clock() - begin) / CLOCKS_PER_SEC;
    cout<<"Elapsed time: "<<bTime<<"s"<<endl;
    cout<<"Gain: "<<aTime/bTime<<endl;
    cout<<"Check: "<<(a[SIZE-1][SIZE-1] + b[SIZE-1][SIZE-1])<<endl;
}
#@@END tarea0/tarea0_matriz.cpp
#@@BEGIN tarea0/cambios_tarea0.txt
TAREA 0 APLICADA AL PROGRAMA kNN: QUE SE HA CAMBIADO (Y QUE NO) EN LA BIBLIOTECA
==============================================================================

La Tarea 0 del enunciado ensena dos ideas:
  (1) un std::vector que crece con push_back se reasigna y copia varias veces; si se conoce el
      tamano, es mas rapido reservarlo (reserve / constructor con tamano) y escribir directamente;
  (2) una matriz se guarda por filas, asi que recorrerla por filas (posiciones contiguas) aprovecha
      la cache y por columnas no.
Los programas tarea0_vector.cpp y tarea0_matriz.cpp (resultados en tarea0/resumen_tarea0.txt)
miden ambos efectos con -O0 y -O2 y distintos tamanos.

Se han buscado esas mismas situaciones en el codigo del profesor y se han corregido SOLO las que
no cambian el resultado. Las versiones originales estan en la carpeta del proyecto; las limpias se
extraen en mpi_src/opt/ y se compilan con el sufijo _opt (knn_seq_opt, knn_mpi_opt, ...). Asi
se pueden comparar los dos sabores (SEQ_FLAVORS / MPI_FLAVORS en run_all.sh).

CAMBIOS (todos reducen "gastos generales"; ninguno altera el calculo)
  knn.h
     - El constructor de KNN copiaba el vector de entrenamiento muestra a muestra con push_back
       (el vector crecia y se recolocaba). Ahora lo copia de golpe en la lista de inicializacion.
       Idea (1) de la Tarea 0.
  knn.cc
     - classify(): el vector de resultados (uno por muestra de entrenamiento) se reserva una vez
       y se reutiliza entre llamadas, en vez de crecer con push_back en cada clasificacion.
       Idea (1) de la Tarea 0.
     - maxVotesLabel(): recibe el std::map por referencia constante (antes se copiaba entero en
       cada llamada).
     - classifyAndEvaluate(): se elimina el cout + flush por CADA muestra de test (salida por
       pantalla dentro del bucle caliente; con MPI ademas mezclaria la salida de los procesos) y
       se reserva el vector de etiquetas.
  crossValidation.cc / crossValidation.h
     - splitDataset() reserva el tamano final de los vectores de entrenamiento y validacion (se
       llama una vez por cada par (k, fold)). El constructor asigna el dataset directamente.
     - Se quitan comentarios de documentacion (no afecta al codigo).

LO QUE NO SE HA CAMBIADO A PROPOSITO
  - El calculo de la distancia (se mantiene el std::sqrt).
  - El std::sort de TODAS las distancias. Sustituirlo por una seleccion parcial (nth_element /
    partial_sort) sera mucho mas rapido, pero NO es una limpieza: cuando hay empates de distancia
    el orden de los candidatos cambia y puede cambiar el K elegido o la accuracy. Se deja fuera para
    poder demostrar que TODAS las versiones dan exactamente el mismo resultado que el original.
    Es una optimizacion algoritmica que conviene mencionar en la memoria como trabajo futuro.
  - El recorrido de las imagenes ya es contiguo (imagen aplanada, de principio a fin): no hay
    recorridos por columnas que corregir (idea (2) de la Tarea 0 ya se cumple).

COMPROBACION
  verificacion.txt demuestra que knn_seq, knn_seq_opt y todas las versiones MPI (de ambos
  sabores) producen las mismas lineas "K: ... Accuracy", "Best k" y "Test accuracy". La ganancia
  esperada es pequena (segun gprof, ~75-80 % del tiempo esta en euclidean() y el resto en la
  ordenacion y la copia de muestras, que no se tocan); sirve
  para tener una linea base secuencial limpia y para no paralelizar sobrecostes que se pueden
  eliminar sin mas.

A continuacion se incluye la diferencia exacta (diff -u) entre cada fichero original y el limpio.
==============================================================================
#@@END tarea0/cambios_tarea0.txt
#@@BEGIN grafos/secuencial.dot
// Grafo de control de flujo de la solucion SECUENCIAL (main.cc / knn.cc / crossValidation.cc)
digraph secuencial {
  rankdir=TB; nodesep=0.35; ranksep=0.4;
  node [shape=box, fontname="Helvetica", fontsize=10, style=rounded];
  edge [fontname="Helvetica", fontsize=9];

  start  [shape=oval, label="main"];
  load   [label="loadDataset\n60000 entrenamiento + 10000 test\n(28x28 -> 14x14, normalizadas)"];
  subset [label="subconjunto de ajuste\n(primeras 20000 muestras)"];
  start -> load -> subset;

  subgraph cluster_tuning {
    label="fineTuning: K = 1..10, 5 folds   (~84 % del tiempo)"; style=dashed; fontname="Helvetica"; fontsize=11;
    kloop    [shape=diamond, style="", label="para cada K\n(1..10)"];
    foldloop [shape=diamond, style="", label="para cada fold\n(0..4)"];
    split    [label="splitDataset\nentrenamiento 16000 / validacion 4000"];
    build    [label="KNN(entrenamiento, K)\n(copia las 16000 muestras)"];
    sloop    [shape=diamond, style="", label="para cada muestra\nde validacion (4000)"];
    classify [label="classify(muestra)\n1) distancia euclidea a las 16000\n2) std::sort de las 16000\n3) votos de los K primeros -> moda"];
    acc      [label="accuracy del fold"];
    mean     [label="accuracy(K) = media de los 5 folds"];
    best     [label="si accuracy(K) > mejor:\nmejor K = K"];
    kloop -> foldloop -> split -> build -> sloop;
    sloop -> classify [label="siguiente"];
    classify -> sloop;
    sloop -> acc [label="fin"];
    acc -> foldloop [label="siguiente fold"];
    foldloop -> mean [label="fin de folds"];
    mean -> best;
    best -> kloop [label="siguiente K"];
  }
  subset -> kloop;

  final [label="evaluacion final   (~16 % del tiempo)\nKNN(60000 entrenamiento, mejor K)\nclassify de las 10000 muestras de test"];
  best -> final [label="fin de K\nDEPENDENCIA: necesita el mejor K", color=red, fontcolor=red];
  out [shape=oval, label="Best k, Test accuracy"];
  final -> out;

  n1 [shape=note, style="", fontsize=9, label="INDEPENDIENTES:\n- las 50 tareas (K, fold) entre si\n- las muestras de una validacion entre si\n- las muestras de test entre si"];
  n2 [shape=note, style="", fontsize=9, label="DEPENDENCIAS:\n- media de folds necesita los 5 folds\n- mejor K necesita todos los K\n- evaluacion final necesita el mejor K\n  (se rompe con ejecucion especulativa)"];
  n1 -> classify [style=dotted, arrowhead=none];
  n2 -> final [style=dotted, arrowhead=none];
}
#@@END grafos/secuencial.dot
#@@BEGIN grafos/paso1_datos.dot
// Solucion paralela PASO 1: paralelismo de datos (SPMD, memoria distribuida, paso de mensajes)
digraph paso1 {
  rankdir=TB; nodesep=0.4; ranksep=0.45; compound=true;
  node [shape=box, fontname="Helvetica", fontsize=10, style=rounded];
  edge [fontname="Helvetica", fontsize=9];

  init [shape=oval, label="MPI_Init / MPI_Comm_rank / MPI_Comm_size"];

  subgraph cluster_p0 { label="proceso 0"; style=dashed; fontname="Helvetica";
    l0 [label="carga del dataset\n(replicada)"]; t0 [label="ajuste: tareas t = 0, P, 2P, ...\n(reparto ciclico)"];
    k0 [label="elige mejor K\n(mismo en todos)"]; e0 [label="evaluacion final:\nbloque 0 del test"];
    l0 -> t0; t0 -> k0 [style=invis]; k0 -> e0;
  }
  subgraph cluster_p1 { label="proceso 1"; style=dashed; fontname="Helvetica";
    l1 [label="carga del dataset\n(replicada)"]; t1 [label="ajuste: tareas t = 1, P+1, ...\n(reparto ciclico)"];
    k1 [label="elige mejor K\n(mismo en todos)"]; e1 [label="evaluacion final:\nbloque 1 del test"];
    l1 -> t1; t1 -> k1 [style=invis]; k1 -> e1;
  }
  subgraph cluster_pn { label="proceso P-1"; style=dashed; fontname="Helvetica";
    ln [label="carga del dataset\n(replicada)"]; tn [label="ajuste: tareas t = P-1, 2P-1, ...\n(reparto ciclico)"];
    kn [label="elige mejor K\n(mismo en todos)"]; en [label="evaluacion final:\nbloque P-1 del test"];
    ln -> tn; tn -> kn [style=invis]; kn -> en;
  }

  init -> l0; init -> l1; init -> ln;

  allr [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Allreduce(SUM) de las 50 accuracies\nSINCRONIZACION: todos esperan al mas lento\n(cada proceso aporta solo sus posiciones; el resto 0.0)"];
  t0 -> allr; t1 -> allr; tn -> allr;
  allr -> k0; allr -> k1; allr -> kn;

  red [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Reduce(SUM) de los aciertos -> proceso 0"];
  e0 -> red; e1 -> red; en -> red;
  out [shape=oval, label="proceso 0: Best k, Test accuracy\nMPI_Reduce(MAX) de tiempos + MPI_Gather de metricas"];
  red -> out;
}
#@@END grafos/paso1_datos.dot
#@@BEGIN grafos/paso2_3_especulativo.dot
// Solucion paralela PASOS 2 y 3: paralelismo funcional + ejecucion especulativa
digraph especulativo {
  rankdir=LR; nodesep=0.4; ranksep=0.7; compound=true;
  node [shape=box, fontname="Helvetica", fontsize=10, style=rounded];
  edge [fontname="Helvetica", fontsize=9];

  subgraph cluster_T { label="GRUPO DE AJUSTE (procesos 0 .. nT-1)"; style=dashed; fontname="Helvetica"; fontsize=11;
    ta [label="proceso de ajuste\nestatico: tareas (k, fold) ciclicas\ndinamico: pide una tarea, la calcula,\nenvia el resultado, pide otra"];
  }

  subgraph cluster_E { label="GRUPO DE EVALUACION (procesos nT .. P-1; lider = nT)  -  comunicador propio (MPI_Comm_split)"; style=dashed; fontname="Helvetica"; fontsize=11;
    lead [label="LIDER\n1) clasifica su bloque de test con el\n   mejor K conocido (K inicial supuesto)\n2) cada 25 muestras: MPI_Iprobe\n3) tracker: cuando un K completa sus\n   folds, es el mejor hasta ahora?\n4) avisa de cada cambio de K"];
    work [label="procesos de evaluacion\n1) clasifican su bloque con el K vigente\n2) cada 25 muestras: MPI_Iprobe\n3) K nuevo -> DESCARTAN y REINICIAN\n4) K definitivo + bloque hecho -> fin"];
    help [shape=note, style="", label="MODO DINAMICO (paso 3):\nsi el bloque esta hecho y el ajuste\nno ha terminado, AYUDAN pidiendo\ntareas de ajuste al lider"];
    work -> help [style=dotted, arrowhead=none];
  }

  ta -> lead [label="MPI_Isend {tarea, accuracy}\n(TAG_RESULT, cada vez que acaba una tarea)", color="#1f77b4", fontcolor="#1f77b4"];
  lead -> ta [label="(dinamico) TAG_TASK {tarea | -1}\nrespuesta a TAG_REQ", style=dashed, color="#1f77b4", fontcolor="#1f77b4"];
  lead -> work [label="MPI_Isend {K, definitivo?}\n(TAG_BEST, cuando mejora el K;\nel ultimo con definitivo = 1)", color="#d62728", fontcolor="#d62728"];
  work -> lead [label="(dinamico) TAG_REQ y\nTAG_RESULT de la ayuda", style=dashed, color="#2ca02c", fontcolor="#2ca02c"];

  bar [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Ibarrier final (sin gastar CPU;\nel lider sigue atendiendo peticiones)"];
  red [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Reduce(SUM) de aciertos\nen el comunicador del grupo\n+ MPI_Gather de metricas -> lider"];
  ta -> bar; lead -> bar; work -> bar;
  bar -> red;
  out [shape=oval, label="lider: Best k, Test accuracy,\nevolucion del mejor K, reinicios,\ntrabajo descartado, CSV"];
  red -> out;

  det [shape=note, style="", fontsize=9, label="K definitivo = mayor accuracy;\nen empate, el K menor\n(igual que el secuencial).\nResultado identico al secuencial."];
  det -> lead [style=dotted, arrowhead=none];
}
#@@END grafos/paso2_3_especulativo.dot
__PAYLOAD_NO_EJECUTAR__
