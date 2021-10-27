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
        cmdParser.addParameterInt("-rz", "Radius of the filter in the z direction ", 2);
        cmdParser.addParameterInt("-rt", "Radius of the filter in the z direction ", 2);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", false);

        cmdParser.addParameterBoolean("-verbose", "Print progress to console", true);
        cmdParser.setMan("Denoise a 3D+t image with the Median filter");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const int rx = cmdParser.getParameterInt("-rx");
        const int ry = cmdParser.getParameterInt("-ry");
        const int rz = cmdParser.getParameterInt("-rz");
        const int rt = cmdParser.getParameterInt("-rt");
        const bool verbose = cmdParser.getParameterBool("-verbose");
        const bool padding = cmdParser.getParameterBool("-padding");

        if (inputImageFile == ""){
            observer->message("median4d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("median4d: input image: " + inputImageFile);
            observer->message("median4d: output image: " + outputImageFile);
            observer->message("median4d: rx: " + std::to_string(rx));
            observer->message("median4d: ry: " + std::to_string(ry));
            observer->message("median4d: rz: " + std::to_string(rz));
            observer->message("median4d: rt: " + std::to_string(rt));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        unsigned int st = inputImage->getSizeT();
        if (inputImage->getSizeC() > 1)
        {
            throw SException("median4d can process only 3D+t gray scale images");
        }

        SObservable * observable = new SObservable();
        observable->addObserver(observer);
        SImg::tic();

        if (padding){
            if (verbose){
                observer->message("median4d: use padding");
            }

            // padding 
            unsigned int sx_pad = sx + 2*rx;
            unsigned int sy_pad = sy + 2*ry;
            unsigned int sz_pad = sz + 2*rz;
            unsigned int st_pad = st;
            float* noisy_padded_image = new float[sx_pad*sy_pad*sz_pad*st_pad];
            SImg::mirror_padding_4d(noisy_image, noisy_padded_image, sx, sy, sz, st, sx_pad, sy_pad, sz_pad);

            // median filtering
            float *output_pad = new float[sx_pad*sy_pad*sz_pad*st_pad];
            SImg::medianFilter(noisy_padded_image, sx_pad, sy_pad, sz_pad, st_pad, 1, rx, ry, rz, rt, output_pad);
            delete[] noisy_padded_image;

            // remove padding
            float* output = new float[sx*sy*sz*st];
            SImg::remove_padding_4d(output_pad, output, sx_pad, sy_pad, sz_pad, st_pad, sx, sy, sz);
            delete[] output_pad;

            SImg::toc();
            delete STimerAccess::instance();

            SImageFloat* denImage = new SImageFloat(output, sx, sy, sz, st);
            SImageReader::write(denImage, outputImageFile);
            delete inputImage;
            delete denImage;
        }
        else{
            float *output = new float[sx*sy*sz*st];
            SImg::medianFilter(noisy_image, sx, sy, sz, st, 1, rx, ry, rz, rt, output);
            SImg::toc();
            delete STimerAccess::instance();
            
            SImageFloat* denImage = new SImageFloat(output, sx, sy, sz, st);
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
