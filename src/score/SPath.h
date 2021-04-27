/// \file SPath.h
/// \brief SPath
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "scoreExport.h"

/// \class SPath
/// \brief Operation of on path strings
class SCORE_EXPORT SPath
{
public:
    /// \fn SPath();
    /// \brief Constructor();
    SPath();

public:
    /// \fn static std::string getFileNameFromPath(std::string path);
    /// \brief extract a filename for a path
    /// \param[in] path String containing the path to analyse
    /// \return a string containing the path without filename
    static std::string getFileNameFromPath(std::string path);
    /// \fn static std::string removeFileNameFromPath(std::string file);
    /// \brief Remove the filename from a full path to get the parent directory full path
    /// \param[in] file String containing the filename to analyse
    /// \return a string containing the parent folder full path
    static std::string removeFileNameFromPath(std::string file);
    /// \fn static bool isPath(std::string path);
    /// \brief Test if a string is a path 
    /// The test is done by checking if it contains slash of backslash
    static bool isPath(std::string path);
    /// \fn static std::string join(std::string path, std::string filename)
    /// \brief Join a filname to it directory path
    /// \param[in] path String containing the path
    /// \param[in] filename String containing the filename
    /// \return a string containing the file full path
    static std::string join(std::string path, std::string filename);
    /// \fn static std::string relativeToFilename(std::string filename, std::string referenceFileName);
    /// \brief construct a filename full from a filename and a reference filename
    /// \param[in] filename Filename we want to retrieve the full path
    /// \param[in] referenceFileName Full path of a file in the same dir
    /// \return the reconstruct path if it exists. Otherwise return an empty string
    static std::string relativeToFilename(std::string filename, std::string referenceFileName);
};
