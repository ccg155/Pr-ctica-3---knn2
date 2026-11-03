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
