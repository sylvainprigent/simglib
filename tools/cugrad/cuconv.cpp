#include <score>
#include <scli>
#include <simageio>
#include <cugrad>
#include "math.h"

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        // Parse inputs
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addInputData("-k", "Convolution kernel image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Convolution with cuda fft");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string kernelImageFile = cmdParser.getDataURI("-k");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("cuconv: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("cuconv: input image: " + inputImageFile);
            observer->message("cuconv: input image: " + kernelImageFile);
            observer->message("cuconv: output image: " + outputImageFile);
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        SImageFloat* kernelImage = dynamic_cast<SImageFloat*>(SImageReader::read(kernelImageFile, 32));
        float* image = inputImage->getBuffer();
        float* kernel = kernelImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        if (inputImage->getSizeZ() > 1 || inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("cuconv can process only 2D gray scale images");
        }

        float *output = (float*)malloc(sx*sy*sizeof(float));
        
        SImg::convolve2d(image, kernel, sx, sy, output);

        SImageReader::write(new SImageFloat(output, sx, sy), outputImageFile);

        delete inputImage;
        delete[] output;
        //delete observable;
    }
    catch (SException &e)
    {
        observer->message(e.what(), SObserver::MessageTypeError);
        return 1;
    }
    catch (std::exception &e)
    {
        observer->message(e.what(), SObserver::MessageTypeError);
        return 1;
    }

    delete observer;
    return 0;
}
