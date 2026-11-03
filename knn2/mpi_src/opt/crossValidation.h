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
