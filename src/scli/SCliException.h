/// \file SException.h
/// \brief SException
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <iostream>
#include <sstream>
#include <exception>

#include "scliExport.h"

/// \class SException
/// \brief class defining the exeptions used in the code
/// heritate from the std::exception
class SCLI_EXPORT SCliException : public std::exception
{
public:
    /// \fn SException( const char * Msg )
    /// \brief Constructor
    /// \param[in] Msg Message
    SCliException( const char * Msg )
    {
        std::ostringstream oss;
        oss << "" << Msg; // print Error ?
        this->msg = oss.str();
    }

    /// \fn virtual ~blException() throw()
    /// \brief Desctructor
    virtual ~SCliException() throw()
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
