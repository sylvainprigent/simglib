/// \file SSingleton.h
/// \brief SSingleton
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <stddef.h>

/// \class SSingleton
/// \brief templated singleton patern that can be applied to any class
template <typename T>
class SSingleton
{
protected:
    /// \fn SSingleton();
    /// \brief Constructor
    SSingleton(){}
    /// \fn ~SSingleton();
    /// \brief Destructor
    ~SSingleton(){}

public:
    // Publique interace
    /// \brief Instanciate the singleton
    /// \return an instance of the templated class
    static T *instance ()  {
        if (m_singleton == NULL)
        {
            //std::cout << "creating singleton." << std::endl;
            m_singleton = new T;
        }
        else
        {
            //std::cout << "singleton already created!" << std::endl;
        }

        return (static_cast<T*> (m_singleton));
    }

    /// \brief free the instance of the class
    static void kill(){
      if (NULL != m_singleton)
        {
          delete m_singleton;
          m_singleton = NULL;
        }
    }

private:
    static T *m_singleton; ///< Unique instance
};

template <typename T>
T *SSingleton<T>::m_singleton = NULL;
