/// \file SHistogram.cpp
/// \brief SHistogram class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SHistogram.h"

SHistogram::SHistogram(){
    m_processName = "SHistogram";
    m_processPrecision = 32;
    m_processZ = true;
    m_processT = false;
    m_processC = false;

    m_numberOfBins = 256;
}

void SHistogram::setNumberOfBins(unsigned numberOfBins){
    m_numberOfBins = numberOfBins;
}

unsigned SHistogram::NumberOfBins(){
    return m_numberOfBins;
}

int* SHistogram::getCount(){
    return m_count;
}

float* SHistogram::getValues(){
    return m_values;
}

int SHistogram::countAt(int idx){
    return m_count[idx];
}

float SHistogram::valueAt(int idx){
    return m_values[idx];
}

void SHistogram::run(){

    SImageFloat* input = this->castInputToFloat();

    m_count = new int[m_numberOfBins];
    m_values = new float[m_numberOfBins];

    for (unsigned int i = 0 ; i < m_numberOfBins ; i++){
        m_count[i] = 0;
    }

    int size =int(input->getSizeX()*input->getSizeY()*input->getSizeZ()*input->getSizeC());
    float *buffer = input->getBuffer();
    float vmin = input->getMin();
    float vmax = input->getMax();
    float val;
    unsigned int idx;
    for (int i = 0 ; i < size ; i++){
        val = buffer[i];
        idx = unsigned(((val - vmin)/(vmax - vmin))*(m_numberOfBins-1));
        m_count[idx] += 1;
    }
    for (unsigned int i = 0 ; i < m_numberOfBins ; i++){
        m_values[i] = vmin + float(i)*( (vmax-vmin)/float(m_numberOfBins) );
    }

}

STable* SHistogram::toTable(){

    STable* table = new STable(m_numberOfBins, 2);
    table->setHeader(0, "values");
    table->setHeader(1, "count");

    for (unsigned int i = 0 ; i < m_numberOfBins ; i++){
        table->set(i,0,m_values[i]);
        table->set(i,1,m_count[i]);
    }

    return table;
}
