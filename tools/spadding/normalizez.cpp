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

        cmdParser.setMan("Normalize each z plan in [0, 1]");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        if (inputImageFile == ""){
            observer->message("ZeroPadding: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("Resize: input image: " + inputImageFile);
            observer->message("Resize: output image: " + outputImageFile);
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));


        float* buffer_in = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();

        
        float* buffer_out = new float[sx*sy*sz];
        for (unsigned int z = 0 ; z < sz ; z++){

            float minz = buffer_in[z + sz*(sy*0+0)];
            float maxz = minz;    
            float val;
            for (unsigned int x = 0 ; x < sx ; x++){
                for (unsigned int y = 0 ; y < sy ; y++){
                    val = buffer_in[z + sz*(sy*x+y)];
                    if (val < minz){
                        minz = val;
                    }
                    if (val > maxz){
                        maxz = val;
                    }
                }
            }
            for (unsigned int x = 0 ; x < sx ; x++){
                for (unsigned int y = 0 ; y < sy ; y++){
                    buffer_out[z + sz*(sy*x+y)] = (buffer_in[z + sz*(sy*x+y)] - minz)/(maxz-minz);
                }
            }
        }

        // save outputs
        SImageFloat* outputImage = new SImageFloat(buffer_out, sx, sy, sz);
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
