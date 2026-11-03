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
