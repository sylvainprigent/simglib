/// \file SCliUtils.cpp
/// \brief SCliUtils class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#include "SCliUtils.h"

#include <stdio.h>  /* defines FILENAME_MAX */
#include <string.h>
#if defined(_WIN32)
#define BiOS "windows"
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define BiOS "unix"
#define GetCurrentDir getcwd
#endif
#include<iostream>

std::string SCliUtils::getCurentPath(){
    char buff[FILENAME_MAX];
    //char* v = GetCurrentDir( buff, FILENAME_MAX );
    std::string current_working_dir(buff);
    return current_working_dir;
}

std::string SCliUtils::getFileSeparator(){
    if ( strcmp(BiOS, "windows") == 0){
        return "\\";
    }
    return "/";
}
