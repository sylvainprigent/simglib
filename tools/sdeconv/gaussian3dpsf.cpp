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
        cmdParser.addParameterFloat("-sigmaxy", "PSF width and height", 1.0);
        cmdParser.addParameterFloat("-sigmaz" ,"PSF depth", 1.0);
        cmdParser.addParameterInt("-width" ,"image width", 256);
        cmdParser.addParameterInt("-height" ,"image height", 256);
        cmdParser.addParameterInt("-depth" ,"image depth", 256);
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.setMan("Generate a Gaussian PSF");
        cmdParser.parse(4);

        float sigmaxy = cmdParser.getParameterFloat("-sigmaxy");
        float sigmaz = cmdParser.getParameterFloat("-sigmaz");
        unsigned int sx = cmdParser.getParameterInt("-width");
        unsigned int sy = cmdParser.getParameterInt("-height");
        unsigned int sz = cmdParser.getParameterInt("-depth");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        if (sigmaxy < 0 || sigmaz < 0){
            observer->message("sigma must be positive");
            return 1;
        }

        if (outputImageFile == ""){
            observer->message("Output image file path is empty");
            return 1;
        }

        float* buffer_out = new float[sx*sy*sz];
        SImg::gaussian_psf_3d(buffer_out, sx, sy, sz, sigmaxy, sigmaxy, sigmaz);

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
