/// \file SRandomGeneratorPMS.h
/// \brief SRandomGeneratorPMS class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2014

#pragma once

#include "scoreExport.h"

/// \class SRandomGeneratorPMS
/// \brief “Minimal” random number generator of Park and Miller with Bays-Durham shuffle and added
/// safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint
/// values). Call with idum a negative integer to initialize; thereafter, do not alter idum between
/// successive deviates in a sequence. RNMX should approximate the largest floating value that is
/// less than 1.
class SCORE_EXPORT SRandomGeneratorPMS
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
