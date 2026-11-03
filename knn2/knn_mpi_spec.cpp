// =============================================================================
// knn_mpi_spec.cpp  -  PASO 2: paralelismo FUNCIONAL + ejecucion ESPECULATIVA
//
// Secuencial:  ajuste de K (validacion cruzada)  ->  elegir K  ->  evaluacion final.
// La evaluacion final depende del K que salga del ajuste. Aqui se rompe esa
// dependencia "apostando": la evaluacion final EMPIEZA ya, con el mejor K
// conocido hasta ese momento, mientras el ajuste sigue ejecutandose.
//
// Dos grupos de procesos (MPI_Comm_split) que trabajan A LA VEZ:
//
//   GRUPO DE AJUSTE (rangos 0 .. nT-1)
//     Se reparten las 50 tareas (k, fold) igual que en el paso 1, pero NO hay
//     Allreduce: cada proceso envia el resultado de cada tarea, segun la acaba,
//     al LIDER del grupo de evaluacion (MPI_Isend, tag TAG_RESULT).
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
//                               [entrenamiento_final] [procesos_evaluacion] [k_inicial]
//   por defecto: 10 5 20000 todas todas 0(=automatico) 1
// =============================================================================

#include <mpi.h>

#include "sample.h"
#include "knn.h"
#include "mnist.h"

#include <unistd.h> // usleep

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace knn;
using namespace mnist;

namespace
{

const int TAG_RESULT = 1; // ajuste -> lider de evaluacion: {numero de tarea, accuracy}
const int TAG_BEST = 2;   // lider de evaluacion -> resto de evaluacion: {K, es_definitivo}
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

// Igual que en el paso 1: misma particion que CrossValidation::splitDataset.
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
    int restarts = 0;     // veces que se descarto el trabajo por llegar un K distinto
    double compute = 0.0; // segundos clasificando (incluye el trabajo descartado)
    double wasted = 0.0;  // segundos de trabajo descartado
    double wait = 0.0;    // segundos esperando a que termine el ajuste
    std::vector<TimelineEntry> timeline; // solo el lider
};

// -----------------------------------------------------------------------------
// GRUPO DE AJUSTE: calcula sus tareas y envia cada resultado al lider.
// Devuelve los segundos de calculo.
// -----------------------------------------------------------------------------
double runTuning(const std::vector<TSample> &tuning, unsigned folds, int minK, int numTasks,
                 int myIndex, int nT, int leaderRank)
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
    return compute;
}

// -----------------------------------------------------------------------------
// GRUPO DE EVALUACION: clasifica begin..end con el mejor K conocido y reinicia
// si cambia. Devuelve los aciertos del bloque con el K definitivo (en finalK).
// -----------------------------------------------------------------------------
long long speculativeEvaluation(const std::vector<TSample> &train, const std::vector<TSample> &test,
                                std::size_t begin, std::size_t end, int k0, int leaderRank,
                                bool isLeader, const std::vector<int> &evalWorkers,
                                BestKTracker *tracker, double tStart, EvalStats &stats, int &finalK)
{
    int kWanted = k0;        // el K que ahora mismo se cree el mejor
    bool finalKnown = false; // true cuando el ajuste ha terminado y kWanted es el definitivo

    // Solo el lider: envios de avisos al resto del grupo.
    std::deque<std::array<int, 2>> outBuf; // deque: las referencias no se invalidan al crecer
    std::vector<MPI_Request> outReq;
    auto announce = [&](int k, bool fin)
    {
        for (int w : evalWorkers)
        {
            outBuf.push_back({k, fin ? 1 : 0});
            outReq.emplace_back();
            MPI_Isend(outBuf.back().data(), 2, MPI_INT, w, TAG_BEST, MPI_COMM_WORLD, &outReq.back());
        }
    };

    // Consulta mensajes sin bloquear. Devuelve true si ha procesado alguno.
    auto poll = [&]() -> bool
    {
        bool got = false;
        int flag = 0;
        if (isLeader)
        {
            for (;;)
            {
                MPI_Status st;
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &st);
                if (!flag) break;
                double msg[2];
                MPI_Recv(msg, 2, MPI_DOUBLE, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                got = true;

                const bool changed = tracker->add(static_cast<int>(msg[0]), msg[1]);
                if (tracker->complete())
                {
                    kWanted = tracker->bestK();
                    finalKnown = true;
                    stats.timeline.push_back({MPI_Wtime() - tStart, kWanted, true});
                    announce(kWanted, true);
                }
                else if (changed)
                {
                    kWanted = tracker->bestK();
                    stats.timeline.push_back({MPI_Wtime() - tStart, kWanted, false});
                    announce(kWanted, false);
                }
            }
        }
        else
        {
            for (;;)
            {
                MPI_Iprobe(leaderRank, TAG_BEST, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
                if (!flag) break;
                int msg[2];
                MPI_Recv(msg, 2, MPI_INT, leaderRank, TAG_BEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                got = true;
                kWanted = msg[0];
                finalKnown = (msg[1] != 0);
            }
        }
        return got;
    };

    int kUsed = -1;          // K con el que se esta clasificando de verdad
    long long matches = 0;
    std::size_t i = begin;
    double workThisK = 0.0;  // trabajo hecho con kUsed (se pierde si hay que reiniciar)
    std::unique_ptr<KNN> knn;

    for (;;)
    {
        if (kUsed != kWanted) // (re)inicio: o es la primera vez, o ha llegado un K distinto
        {
            if (kUsed != -1)
            {
                stats.restarts++;
                stats.wasted += workThisK; // lo hecho con el K anterior se descarta
            }
            const double t0 = MPI_Wtime();
            kUsed = kWanted;
            knn.reset(new KNN(train, kUsed));
            matches = 0;
            i = begin;
            workThisK = MPI_Wtime() - t0;
            stats.compute += workThisK;
        }

        if (i < end) // quedan muestras: un trozo de CHUNK y luego miro mensajes
        {
            const double t0 = MPI_Wtime();
            const std::size_t chunkEnd = std::min(i + static_cast<std::size_t>(CHUNK), end);
            for (; i < chunkEnd; i++)
            {
                if (knn->classify(test[i]) == test[i].label)
                    matches++;
            }
            const double dt = MPI_Wtime() - t0;
            workThisK += dt;
            stats.compute += dt;
            poll();
            continue;
        }

        // Bloque terminado con kUsed.
        if (finalKnown && kWanted == kUsed)
            break; // el K usado es el definitivo: hecho

        // Aun puede cambiar el K: espero mensajes sin ocupar CPU.
        const double t0 = MPI_Wtime();
        while (kWanted == kUsed && !finalKnown)
        {
            if (!poll())
                usleep(1000);
        }
        stats.wait += MPI_Wtime() - t0;
    }

    if (isLeader)
        MPI_Waitall(static_cast<int>(outReq.size()), outReq.data(), MPI_STATUSES_IGNORE);

    finalK = kUsed;
    return matches;
}

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
    try
    {
        if (argc > 1) maxK = std::stoul(argv[1]);
        if (argc > 2) folds = std::stoul(argv[2]);
        if (argc > 3) tuningSamples = std::stoul(argv[3]);
        if (argc > 4) testSamples = std::stoul(argv[4]);
        if (argc > 5) finalTrainSamples = std::stoul(argv[5]);
        if (argc > 6) evalProcs = std::stoi(argv[6]);
        if (argc > 7) initialK = std::stoul(argv[7]);
    }
    catch (const std::exception &)
    {
        abortAll(rank, "Uso: knn_mpi_spec [maxK] [folds] [muestras_ajuste] [muestras_test] "
                       "[entrenamiento_final] [procesos_evaluacion] [k_inicial]");
    }
    if (maxK <= minK || folds < 2)
        abortAll(rank, "Se necesita maxK > 1 y folds >= 2");
    if (initialK < minK || initialK > maxK)
        abortAll(rank, "k_inicial debe estar entre 1 y maxK");

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
    double tuneCompute = 0.0;
    long long totalMatches = 0;
    int finalK = -1;
    std::unique_ptr<BestKTracker> tracker;

    if (!isEval)
    {
        tuneCompute = runTuning(tuning, folds, minK, numTasks, rank, nT, leaderRank);
    }
    else
    {
        const int er = rank - nT; // indice dentro del grupo de evaluacion
        const std::size_t begin = nTest * static_cast<std::size_t>(er) / nE;
        const std::size_t end = nTest * static_cast<std::size_t>(er + 1) / nE;

        std::vector<int> workers; // rangos del resto del grupo (solo los usa el lider)
        if (isLeader)
        {
            tracker.reset(new BestKTracker(minK, numK, folds));
            for (int r = nT + 1; r < size; r++)
                workers.push_back(r);
        }

        const long long matches = speculativeEvaluation(*finalTrain, test, begin, end, initialK, leaderRank,
                                                        isLeader, workers, tracker.get(), tStart, stats, finalK);
        MPI_Reduce(&matches, &totalMatches, 1, MPI_LONG_LONG, MPI_SUM, 0, groupComm); // 0 = lider
    }
    const double tDone = MPI_Wtime() - tStart; // cuando termina ESTE proceso su trabajo

    sleepyBarrier(MPI_COMM_WORLD);

    // ---- Recogida de metricas en el lider -----------------------------------
    // {es_evaluacion, fin, calculo, espera, descartado, reinicios, carga}
    const double mine[7] = {isEval ? 1.0 : 0.0, tDone, isEval ? stats.compute : tuneCompute,
                            stats.wait, stats.wasted, static_cast<double>(stats.restarts), tLoad};
    std::vector<double> all(7 * size);
    MPI_Gather(mine, 7, MPI_DOUBLE, all.data(), 7, MPI_DOUBLE, leaderRank, MPI_COMM_WORLD);

    if (isLeader)
    {
        double total = 0.0, maxLoad = 0.0;
        for (int r = 0; r < size; r++)
        {
            total = std::max(total, all[7 * r + 1]);
            maxLoad = std::max(maxLoad, all[7 * r + 6]);
        }

        std::cout << "Procesos MPI: " << size << " (ajuste: " << nT << ", evaluacion: " << nE << ") | K en ["
                  << minK << ", " << maxK << "], " << folds << " folds, " << nTuning << " muestras de ajuste, "
                  << nTest << " de test, " << nFinalTrain << " de entrenamiento final, K inicial " << initialK
                  << std::endl;
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
                  << ", trabajo descartado (lider): " << stats.wasted << " s" << std::endl;

        std::cout << "Por proceso (segundos):" << std::endl;
        for (int r = 0; r < size; r++)
        {
            const double *v = &all[7 * r];
            std::cout << "  proceso " << r << (v[0] > 0.5 ? " [evaluacion]" : " [ajuste]    ")
                      << "  termina a t=" << v[1] << "  calculo=" << v[2];
            if (v[0] > 0.5)
                std::cout << "  descartado=" << v[4] << "  espera=" << v[3] << "  reinicios=" << static_cast<int>(v[5]);
            std::cout << std::endl;
        }
        std::cout << "Carga del dataset (replicada, maximo): " << maxLoad << std::endl;
        std::cout << "Total elapsed time " << total << " seconds" << std::endl;
    }

    MPI_Comm_free(&groupComm);
    MPI_Finalize();
    return 0;
}
