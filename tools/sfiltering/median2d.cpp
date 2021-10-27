#include <score>
#include <scli>
#include <simageio>
#include <sfiltering>
#include <spadding>
#include "math.h"

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        // Parse inputs
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterInt("-rx", "Radius of the filter in the x direction ", 2);
        cmdParser.addParameterInt("-ry", "Radius of the filter in the y direction ", 2);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", true);
 
        cmdParser.addParameterBoolean("-verbose", "Print progress to console", true);
        cmdParser.setMan("Denoise a 2D image with the median filter");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const int rx = cmdParser.getParameterInt("-rx");
        const int ry = cmdParser.getParameterInt("-ry");
        const bool verbose = cmdParser.getParameterBool("-verbose");
        const bool padding = cmdParser.getParameterBool("-padding");

        if (inputImageFile == ""){
            observer->message("median2d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("median2d: input image: " + inputImageFile);
            observer->message("median2d: output image: " + outputImageFile);
            observer->message("median2d: rx: " + std::to_string(rx));
            observer->message("median2d: ry: " + std::to_string(ry));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        if (inputImage->getSizeZ() > 1 || inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("median2d can process only 2D gray scale images");
        }

        SObservable * observable = new SObservable();
        observable->addObserver(observer);
        SImg::tic();

        if (padding){
            if (verbose){
                observer->message("median2d: use padding");
            }
            // padding 
            unsigned int sx_pad = sx + 2*rx;
            unsigned int sy_pad = sy + 2*ry;
            float* noisy_padded_image = new float[sx_pad*sy_pad];
            SImg::mirror_padding_2d(noisy_image, noisy_padded_image, sx, sy, sx_pad, sy_pad);

            // median filtering
            float *output_pad = new float[sx_pad*sy_pad];
            SImg::medianFilter(noisy_padded_image, sx_pad, sy_pad, 1, 1, 1, rx, ry, 0, 0, output_pad);
            delete[] noisy_padded_image;

            // remove padding
            float* output = new float[sx*sy];
            SImg::remove_padding_2d(output_pad, output, sx_pad, sy_pad, sx, sy);
            delete[] output_pad;

            SImg::toc();
            delete STimerAccess::instance();

            SImageFloat* denImage = new SImageFloat(output, sx, sy);
            SImageReader::write(denImage, outputImageFile);
            delete inputImage;
            delete denImage;
        }
        else{
            float *output = new float[sx*sy];
            SImg::medianFilter(noisy_image, sx, sy, 1, 1, 1, rx, ry, 0, 0, output);
            SImg::toc();
            delete STimerAccess::instance();

            SImageFloat* denImage = new SImageFloat(output, sx, sy);
            SImageReader::write(denImage, outputImageFile);
            delete inputImage;
            delete denImage;
        }
        delete observable;
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
