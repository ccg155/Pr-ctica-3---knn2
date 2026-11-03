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
