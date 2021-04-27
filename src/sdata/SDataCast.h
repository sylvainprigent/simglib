/// \file SDataCast.h
/// \brief SDataCast class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include "sdataExport.h"

#include "SData.h"
#include "SInt.h"
#include "SFloat.h"
#include "SString.h"
#include "SObject.h"
#include "SArray.h"

/// \class SDataCast
/// \brief Static functions to cast SData and throw axception in case of error
class SDATA_EXPORT SDataCast{

public:
    static SInt* toSInt(SData* data);
    static SFloat* toSFloat(SData* data);
    static SString* toSString(SData* data);
    static SObject* toSObject(SData* data);
    static SArray* toSArray(SData* data);

    static int toInt(SData* data);
    static float toFloat(SData* data);
    static std::string toString(SData* data);
};
