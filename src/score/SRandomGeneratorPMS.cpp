/// \file SRandomGeneratorPMS.cpp
/// \brief SRandomGeneratorPMS class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2014

#include "SRandomGeneratorPMS.h"

#define IA 16807 ///< IA
#define IM 2147483647 ///< IM
#define AM (1.0/IM) ///< AM
#define IQ 127773 ///< IQ
#define IR 2836 ///< IR
#define NTAB 32 ///< NTAB
#define NDIV (1+(IM-1)/NTAB) ///< NDIV
#define EPS 1.2e-7 ///< espilon
#define RNMX (1.0-EPS) ///< RNMX

long SRandomGeneratorPMS::m_seed;

float SRandomGeneratorPMS::rand(){

    int j;
    long k;
    static long iy=0;
    static long iv[NTAB];
    float temp;
    if (m_seed <= 0 || !iy) { //Initialize.
        if (-(m_seed) < 1) m_seed=1; //Be sure to prevent idum = 0.
        else m_seed = -(m_seed);
        for (j=NTAB+7;j>=0;j--) { //Load the shuffle table (after 8 warm-ups).
            k=(m_seed)/IQ;
            m_seed=IA*(m_seed-k*IQ)-IR*k;
            if (m_seed < 0) m_seed += IM;
            if (j < NTAB) iv[j] = m_seed;
        }
        iy=iv[0];
    }
    k=(m_seed)/IQ; // Start here when not initializing.
    m_seed=IA*(m_seed-k*IQ)-IR*k; //Compute idum=(IAm_seed) % IM //without overif
    if (m_seed < 0) m_seed += IM; // flows by Schrage’s method.
    j=iy/NDIV; // Will be in the range 0..NTAB-1.
    iy=iv[j]; // Output previously stored value and refill the
    iv[j] = m_seed; //shuffle table.
    if ((temp=float(AM)*iy) > float(RNMX)) return float(RNMX); //Because users don’t expect endpoint values.
    else return temp;
}

void SRandomGeneratorPMS::srand(long seed){
    m_seed = seed;
}
