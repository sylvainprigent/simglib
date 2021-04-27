/// \file SDataCast.cpp
/// \brief SDataCast class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SDataCast.h"
#include "SDataException.h"

SInt* SDataCast::toSInt(SData* data){
    SInt* dataObj = dynamic_cast<SInt*>(data);
    if (dataObj){
        return dataObj;
    }
    throw SDataException("Unable to cast SData to SInt");
}

SFloat* SDataCast::toSFloat(SData* data){
    SFloat* dataObj = dynamic_cast<SFloat*>(data);
    if (dataObj){
        return dataObj;
    }
    throw SDataException("Unable to cast SData to SFloat");
}

SString* SDataCast::toSString(SData* data){
    SString* dataObj = dynamic_cast<SString*>(data);
    if (dataObj){
        return dataObj;
    }
    throw SDataException("Unable to cast SData to SString");
}

SObject* SDataCast::toSObject(SData* data){
    SObject* dataObj = dynamic_cast<SObject*>(data);
    if (dataObj){
        return dataObj;
    }
    throw SDataException("Unable to cast SData to SObject");
}

SArray* SDataCast::toSArray(SData* data){
    SArray* dataObj = dynamic_cast<SArray*>(data);
    if (dataObj){
        return dataObj;
    }
    throw SDataException("Unable to cast SData to SArray");
}

int SDataCast::toInt(SData* data){
    SInt* obj = SDataCast::toSInt(data);
    return obj->get();
}

float SDataCast::toFloat(SData* data){
    SFloat* dataObj = dynamic_cast<SFloat*>(data);
    if (dataObj){
        return dataObj->get();
    }

    SInt* dataIntObj = dynamic_cast<SInt*>(data);
    if (dataIntObj){
        return float(dataIntObj->get());
    }

    throw SDataException("Unable to cast SData to SFloat");
}

std::string SDataCast::toString(SData* data){
    SString* obj = SDataCast::toSString(data);
    return obj->get();
}
