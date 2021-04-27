#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <spadding>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterInt("-padx", "Padding in the x direction", 12);
        cmdParser.addParameterInt("-pady", "Padding in the y direction", 12);
        cmdParser.addParameterInt("-padz", "Padding in the z direction", 12);
        cmdParser.setMan("Add zero padding around an image");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const int padx = cmdParser.getParameterFloat("-padx");
        const int pady = cmdParser.getParameterFloat("-pady");
        const int padz = cmdParser.getParameterFloat("-padz");

        if (inputImageFile == ""){
            observer->message("ZeroPadding: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("Padding: input image: " + inputImageFile);
            observer->message("Padding: output image: " + outputImageFile);
            observer->message("Padding: padx: " + std::to_string(padx));
            observer->message("Padding: pady: " + std::to_string(pady));
            observer->message("Padding: padz: " + std::to_string(padz));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));


        float* buffer_in = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        unsigned int sx_out = sx-2*padx;
        unsigned int sy_out = sy-2*pady;
        unsigned int sz_out = sz-2*padz;
        float* buffer_out = new float[sx_out*sy_out*sz_out];

        std::cout << "sx = " << sx << std::endl;
        std::cout << "sy = " << sy << std::endl;
        std::cout << "sz = " << sz << std::endl;
        std::cout << "sx_out = " << sx_out << std::endl;
        std::cout << "sy_out = " << sy_out << std::endl;
        std::cout << "sz_out = " << sz_out << std::endl;

        int ctrl = SImg::remove_padding_3d(buffer_in, buffer_out, sx, sy, sz, sx_out, sy_out, sz_out);
        if (ctrl > 0){
            observer->message("ZeroPadding: dimensions missmatch", SObserver::MessageTypeError);
            return 1;
        }

        // save outputs
        SImageFloat* outputImage = new SImageFloat(buffer_out, sx_out, sy_out, sz_out);
        SImageReader::write(outputImage, outputImageFile);

        delete inputImage;
        delete outputImage;
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
