/// \file STxtIO.h
/// \brief STxtIO class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "simageioExport.h"

#include "SImageIO.h"
#include "STiffIO.h"

#include "simage/SImageFloat.h"
#include "simage/SImageUInt.h"
#include "simage/SImageInt.h"

/// \class STxtIO
/// \brief Read and write images using libtiff
class SIMAGEIO_EXPORT STxtIO : public SImageIO{

public:
    STxtIO();

public:
    SImage* read(std::string file, char precision);
    SImage* read_chunk(std::string file, char precision, int start_t = -1, int end_t = -1);
    void write(SImage* image, std::string file);
       
private:
    std::vector<std::string> read_frames_paths(const std::string& file);
    void write_output_txt(const std::string& file, const std::vector<std::string>& frames_names);
    std::string construct_output_frame_file(std::string filename);
    std::string num_to_string(unsigned int t, unsigned int maxt);

private:
    std::string m_input_dir_path;
    std::vector<std::string> m_input_filenames;
};

/// \class STxtIOChunk
/// \brief Read and write images using libtiff
class SIMAGEIO_EXPORT STxtIOChunk{

public:
    /// \param file Input image (txt) file 
    STxtIOChunk(const std::string& file, const int& chunkSize, const int& overlap);

public:
    /// \brief Reset the chunk count
    void reset();

    /// \brief Get if the file have a next chunk
    /// \return true if the file have a next chunk, false otherwise
    bool hasNext();

    /// \brief Read 4D image of the next chunk
    /// \return a pointer to the readed image
    SImage* getNext();

    /// \brief Get the next chunk images files
    /// \return a pointer to the readed image
    std::vector<std::string> getNextPaths();

    void writeCurrent(SImage* image, const std::string file);
    void writeTxt(const std::string file);

protected:
    std::vector<std::string> read_frames_paths(const std::string& file);
    std::string num_to_string(unsigned int t, unsigned int maxt);

protected:
    std::vector<std::string> m_input_filenames;
    std::string m_input_dir_path;
    int m_chunkSize;
    int m_overlap;
    int m_current_chunk_start;
    int m_current_chunk_end;
    bool m_hasNext;
};