#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <sdeconv>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        // Parse inputs
        SCliParser cmdParser(argc, argv);
        cmdParser.addParameterFloat("-sigma", "PSF width and height", 1.0);
        cmdParser.addParameterInt("-width" ,"image width", 256);
        cmdParser.addParameterInt("-height" ,"image height", 256);
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.setMan("Generate a Gaussian PSF");
        cmdParser.parse(4);

        float sigma = cmdParser.getParameterFloat("-sigma");
        unsigned int sx = cmdParser.getParameterInt("-width");
        unsigned int sy = cmdParser.getParameterInt("-height");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        if (sigma < 0){
            observer->message("sigma must be positive");
            return 1;
        }

        if (outputImageFile == ""){
            observer->message("Output image file path is empty");
            return 1;
        }

        float* buffer_out = new float[sx*sy];
        SImg::gaussian_psf_2d(buffer_out, sx, sy, sigma, sigma);

        SImageFloat* image = new SImageFloat(buffer_out, sx, sy);
        SImageReader::write(image, outputImageFile);

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
