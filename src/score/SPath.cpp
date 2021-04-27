/// \file SPath.cpp
/// \brief SPath
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2021

#include "SPath.h"
#include "SStringOp.h"
#include "SFile.h"

#include <sstream>
using namespace std;

string SPath::getFileNameFromPath(string file){

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    char slash='\\';
#else
    char slash='/';
#endif

    stringstream stream(file);
    string readedword;
    while( getline(stream, readedword, slash)){
    }
    return readedword;
}

std::string SPath::removeFileNameFromPath(string file)
{   
    std::string filename = SPath::getFileNameFromPath(file);
    return file.substr(0, file.size()-filename.size());
}

bool SPath::isPath(std::string path)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    std::string slash="\\";
#else
    std:string slash="/";
#endif
    if (path.find(slash) != std::string::npos) 
    {
        return true;
    }
    return false;    
}

std::string SPath::join(std::string path, std::string filename)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    std::string slash="\\";
#else
    std:string slash="/";
#endif  

    if ( SStringOp::endsWith(path, slash) )
    {
        return path + filename;
    }
    else
    {
        return path + slash + filename;
    }
}

std::string SPath::relativeToFilename(std::string filename, std::string referenceFileName)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    std::string slash="\\";
#else
    std:string slash="/";
#endif 

    if (SFile::exists(filename))
    {
        return filename;
    }
    else{
        std::string path = SPath::removeFileNameFromPath(referenceFileName);
        std::string fullFilename = SPath::join(path, filename);
        if (SFile::exists(fullFilename))
        {
            return fullFilename;
        }
        else
        {
            return "";
        }
    }

}
