/// \file SRandomGeneratorPM.h
/// \brief SRandomGeneratorPM class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2014

#pragma once

#include "scoreExport.h"

/// \class SRandomGeneratorPM
/// \brief “Minimal” random number generator of Park and Miller. Returns a uniform random deviate
/// between 0.0 and 1.0. Set or reset idum to any integer value (except the unlikely value MASK)
/// to initialize the sequence; idum must not be altered between calls for successive deviates in
/// a sequence.
/// The period is 2^31-2 = 2.0*10^9
class SCORE_EXPORT SRandomGeneratorPM
{

private:
    static long m_seed; ///< seed of the random generator

public:
    /// \fn static float rand();
    /// \return a random value in [0.0,1.0]
    static float rand();
    /// \fn static void srand(long seed);
    /// \brief set the seed of the random generator
    static void srand(long seed);
};
