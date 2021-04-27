/// \file SDataException.h
/// \brief SDataException
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <iostream>
#include <sstream>
#include <exception>

#include "sdataExport.h"

/// \class SDataException
/// \brief class defining the exeptions used for slData library
/// heritate from the std::exception
class SDATA_EXPORT SDataException : public std::exception
{
public:
    /// \fn SDataException( const char * Msg )
    /// \brief Constructor
    /// \param[in] Msg Message
    SDataException( const char * Msg )
    {
        std::ostringstream oss;
        oss << "" << Msg; // print Error ?
        this->msg = oss.str();
    }

    /// \fn virtual ~blException() throw()
    /// \brief Desctructor
    virtual ~SDataException() throw()
    {

    }

    /// \fn virtual const char * what() const throw()
    /// \return the error message
    virtual const char * what() const throw()
    {
        return this->msg.c_str();
    }

private:
    std::string msg; ///< Error message
};
