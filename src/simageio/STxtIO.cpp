/// \file SImageIO.cpp
/// \brief SImageIO class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "STxtIO.h"
#include "SImageIOException.h"
#include "simage/SImage.h"
#include "score/SPath.h"
#include "score/SException.h"

#include <iostream>
#include <fstream>
#include <string>

STxtIO::STxtIO() : SImageIO()
{
    m_input_dir_path = "";
}

std::vector<std::string> STxtIO::read_frames_paths(const std::string& file)
{
    m_input_dir_path = SPath::removeFileNameFromPath(file);
    m_input_filenames.clear();
    std::fstream newfile;
    newfile.open(file, std::ios::in);
    if (newfile.is_open()){ 
        std::string tp;
        while(getline(newfile, tp)){
            m_input_filenames.push_back(tp);
        }
        newfile.close();
    }

    std::cout << "Readed frames are: " << m_input_filenames.size() << std::endl;
    for (unsigned int i = 0 ; i < m_input_filenames.size() ; ++i){
        std::cout << "\t" << m_input_filenames[i] << std::endl;
    }
    return m_input_filenames;
}

std::string STxtIO::construct_output_frame_file(std::string filename)
{

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    char slash='\\';
#else
    char slash='/';
#endif

    return m_input_dir_path + slash + filename;
}

void STxtIO::write_output_txt(const std::string& file, const std::vector<std::string>& frames_names)
{
    std::ofstream out_file;
    out_file.open(file);
    if (out_file.is_open())
    {

        for (int i = 0 ; i < frames_names.size() ; i++)
        {
            out_file << frames_names[i] << std::endl;
        }
        out_file.close();
    }
    else
    {
        throw(SException(std::string("Cannot open the move txt ouput file for writing: " + file).c_str()));
    }
}

SImage* STxtIO::read(std::string file, char precision)
{
    return this->read_chunk(file, precision, -1, -1);
}

SImage* STxtIO::read_chunk(std::string file, char precision, int start_t, int end_t)
{
    // read the list of files in the txt file
    std::vector<std::string> frames = read_frames_paths(file);

    // read the first frame to get the frame sizes
    STiffIO tiffio;
    SImageFloat* frame = tiffio.read_float(m_input_dir_path+frames[0]);
    unsigned int sx = frame->getSizeX(); 
    unsigned int sy = frame->getSizeY();
    unsigned int sz = frame->getSizeZ();
    unsigned int st = frames.size();
    unsigned int sc = frame->getSizeC();
    delete frame;

    SImageFloat* movie = new SImageFloat(sx, sy, sz, st, sc);
    float* movie_buffer = movie->getBuffer();

    int t0 = start_t;
    if (t0 < 0){
        t0 = 0;
    }
    int t1 = end_t;
    if (t1 < 0)
    {
        t1 = frames.size();
    }
    for(int t = t0 ; t < t1 ; t++)
    {
        SImageFloat* frame = tiffio.read_float(m_input_dir_path+frames[t]);
        float* frame_buffer = frame->getBuffer();

        // test the size compatibility
        if (frame->getSizeX() != sx || frame->getSizeY() != sy || frame->getSizeZ() != sz || frame->getSizeC() != sc)
        {
            delete frame;
            delete movie;
            throw(SException("Movie frame sizes are not compatibles"));
        }    

        for(unsigned int x = 0 ; x < sx ; x++)
        {
            for(unsigned int y = 0 ; y < sy ; y++)
            {
                for(unsigned int z = 0 ; z < sz ; z++)
                {
                    for(unsigned int c = 0 ; c < sc ; c++)
                    {
                        movie_buffer[ c + sc*(t + st*(z + sz*(y + sy*x)))] = frame_buffer[ c + sc*(0 + 1*(z + sz*(y + sy*x)))];
                    }
                }
            }
        }
        delete frame;
    }
    return movie;
}

void STxtIO::write(SImage* image, std::string file)
{
    std::string basename = SPath::getFileNameFromPath(file.substr(0, file.size() - 4));
    std::string dirname = SPath::removeFileNameFromPath(file);
    unsigned int st = image->getSizeT();
    std::vector<std::string> frames_names;
    STiffIO tiffio;
    SImageFloat *imageFloat = dynamic_cast<SImageFloat*>(image);
    for (unsigned int t = 1 ; t <= st ; t++)
    {
        //std::cout << "write frame " << t << std::endl;
        std::string frame_name = basename + "_t" + this->num_to_string(t, st) + ".tif";
        frames_names.push_back(frame_name);
        //std::cout << "save frame to:" << dirname + frame_name << std::endl;
        tiffio.write(imageFloat->getFrame(t-1), dirname + frame_name);
    }
    this->write_output_txt(file, frames_names);
}

std::string STxtIO::num_to_string(unsigned int t, unsigned int maxt)
{
    std::string max_str = std::to_string(maxt);
    std::string t_str = std::to_string(t);
    return std::string(max_str.length() - t_str.length(), '0') + t_str;
}


/* **************************
       STxtIOChunk
****************************/
STxtIOChunk::STxtIOChunk(const std::string& file, const int& chunkSize, const int& overlap)
{
    m_input_filenames = this->read_frames_paths(file);
    m_chunkSize = chunkSize;
    m_overlap = overlap;
    this->reset();
}

std::vector<std::string> STxtIOChunk::read_frames_paths(const std::string& file)
{
    m_input_dir_path = SPath::removeFileNameFromPath(file);
    m_input_filenames.clear();
    std::fstream newfile;
    newfile.open(file, std::ios::in);
    if (newfile.is_open()){ 
        std::string tp;
        while(getline(newfile, tp)){
            m_input_filenames.push_back(tp);
        }
        newfile.close();
    }
    return m_input_filenames;
}

void STxtIOChunk::reset()
{
    m_current_chunk_start = 0;
    m_current_chunk_end = 0;
    m_hasNext = true;
}

bool STxtIOChunk::hasNext()
{
    return m_hasNext;
}

SImage* STxtIOChunk::getNext()
{
    // read the list of files in the txt file
    std::vector<std::string> frames = this->getNextPaths();

    // read the first frame to get the frame sizes
    STiffIO tiffio;
    SImageFloat* frame = tiffio.read_float(m_input_dir_path+frames[0]);
    unsigned int sx = frame->getSizeX(); 
    unsigned int sy = frame->getSizeY();
    unsigned int sz = frame->getSizeZ();
    unsigned int st = frames.size();
    unsigned int sc = frame->getSizeC();
    delete frame;

    // create the movie image
    SImageFloat* movie = new SImageFloat(sx, sy, sz, st, sc);
    float* movie_buffer = movie->getBuffer();

    // read each pixels
    for(int t = 0 ; t < frames.size() ; t++)
    {
        SImageFloat* frame = tiffio.read_float(m_input_dir_path+frames[t]);
        float* frame_buffer = frame->getBuffer();

        // test the size compatibility
        if (frame->getSizeX() != sx || frame->getSizeY() != sy || frame->getSizeZ() != sz || frame->getSizeC() != sc)
        {
            delete frame;
            delete movie;
            throw(SException("Movie frame sizes are not compatibles"));
        }    

        for(unsigned int x = 0 ; x < sx ; x++)
        {
            for(unsigned int y = 0 ; y < sy ; y++)
            {
                for(unsigned int z = 0 ; z < sz ; z++)
                {
                    for(unsigned int c = 0 ; c < sc ; c++)
                    {
                        movie_buffer[ c + sc*(t + st*(z + sz*(y + sy*x)))] = frame_buffer[ c + sc*(0 + 1*(z + sz*(y + sy*x)))];
                    }
                }
            }
        }
        delete frame;
    }
    return movie;
}

void STxtIOChunk::writeCurrent(SImage* image, const std::string file)
{
    std::string basename = SPath::getFileNameFromPath(file.substr(0, file.size() - 4));
    std::string dirname = SPath::removeFileNameFromPath(file);
    unsigned int st = image->getSizeT();
    std::vector<std::string> frames_names;
    STiffIO tiffio;
    SImageFloat *imageFloat = dynamic_cast<SImageFloat*>(image);
    for (unsigned int t = 0 ; t < st ; t++)
    {
        //std::cout << "write frame " << t << std::endl;
        std::string frame_name = basename + "_t" + this->num_to_string(m_current_chunk_start+t+1, m_input_filenames.size()) + ".tif";
        frames_names.push_back(frame_name);
        //std::cout << "save frame to:" << dirname + frame_name << std::endl;
        tiffio.write(imageFloat->getFrame(t), dirname + frame_name);
    }
}

void STxtIOChunk::writeTxt(const std::string file)
{
    std::string basename = SPath::getFileNameFromPath(file.substr(0, file.size() - 4));
    std::vector<std::string> frames_names;
    unsigned int st = m_input_filenames.size();
    for (unsigned int t = 0 ; t < st ; t++)
    {
        std::string frame_name = basename + "_t" + this->num_to_string(t+1, st) + ".tif";
        frames_names.push_back(frame_name);
    }

    std::ofstream out_file;
    out_file.open(file);
    if (out_file.is_open())
    {

        for (int i = 0 ; i < frames_names.size() ; i++)
        {
            out_file << frames_names[i] << std::endl;
        }
        out_file.close();
    }
    else
    {
        throw(SException(std::string("Cannot open the move txt ouput file for writing: " + file).c_str()));
    }
}

std::vector<std::string> STxtIOChunk::getNextPaths()
{
    int chunk_start = m_current_chunk_end - m_overlap;
    if (chunk_start < 0)
    {
        chunk_start = 0;
    }

    int chunk_end = chunk_start + m_chunkSize -1;
    if (chunk_end >= m_input_filenames.size())
    {
        chunk_end = m_input_filenames.size()-1;
        m_hasNext = false;
    }

    m_current_chunk_start = chunk_start;
    m_current_chunk_end = chunk_end;

    std::vector<std::string> chunk_files;
    for(int t = m_current_chunk_start ; t <= m_current_chunk_end ; t++)
    {
        chunk_files.push_back(m_input_filenames[t]);
    }
    return chunk_files;
}

std::string STxtIOChunk::num_to_string(unsigned int t, unsigned int maxt)
{
    std::string max_str = std::to_string(maxt);
    std::string t_str = std::to_string(t);
    return std::string(max_str.length() - t_str.length(), '0') + t_str;
}