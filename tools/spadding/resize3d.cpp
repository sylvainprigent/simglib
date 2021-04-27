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

        cmdParser.addParameterInt("-sx", "Ouptut size in the x direction", 511);
        cmdParser.addParameterInt("-sy", "Ouptut size in the y direction", 511);
        cmdParser.addParameterInt("-sz", "Ouptut size in the y direction", 511);
        cmdParser.addParameterInt("-ox", "Offset in the x direction", 0);
        cmdParser.addParameterInt("-oy", "Offset size in the y direction", 0);
        cmdParser.addParameterInt("-oz", "Offset size in the y direction", 0);
        cmdParser.setMan("Add zero padding around an image");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const int sx_out = cmdParser.getParameterFloat("-sx");
        const int sy_out = cmdParser.getParameterFloat("-sy");
        const int sz_out = cmdParser.getParameterFloat("-sz");

        const int ox = cmdParser.getParameterFloat("-ox");
        const int oy = cmdParser.getParameterFloat("-oy");
        const int oz = cmdParser.getParameterFloat("-oz");

        if (inputImageFile == ""){
            observer->message("ZeroPadding: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("Resize: input image: " + inputImageFile);
            observer->message("Resize: output image: " + outputImageFile);
            observer->message("Resize: sx: " + std::to_string(sx_out));
            observer->message("Resize: sy: " + std::to_string(sy_out));
            observer->message("Resize: sz: " + std::to_string(sz_out));
            observer->message("Resize: ox: " + std::to_string(ox));
            observer->message("Resize: oy: " + std::to_string(oy));
            observer->message("Resize: oz: " + std::to_string(oz));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));


        float* buffer_in = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();

        if (ox+sx_out > sx || oy+sy_out > sy || oz+sz_out > sz){
            observer->message("Dimensions not correct", SObserver::MessageTypeError);
            return 1;
        }
        float* buffer_out = new float[sx_out*sy_out*sz_out];
        for (unsigned int z = 0 ; z < sz_out ; z++){
            for (unsigned int x = 0 ; x < sx_out ; x++){
                for (unsigned int y = 0 ; y < sy_out ; y++){
                    buffer_out[z + sz_out*(sy_out*x+y)] = buffer_in[z+oz + sz*(sy*(x+ox)+(y+oy))];
                }
            }
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
