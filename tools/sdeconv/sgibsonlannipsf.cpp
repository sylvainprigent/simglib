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
        cmdParser.addParameterInt("-width" ,"image width", 256);
        cmdParser.addParameterInt("-height" ,"image height", 256);
        cmdParser.addParameterInt("-depth" ,"image depth", 256);
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterFloat("-wavelength" ,"Exitation wavelength (nm)", 610);
        cmdParser.addParameterFloat("-psxy" ,"Pixel size in XY (nm)", 100);
        cmdParser.addParameterFloat("-psz" ,"Pixel size in Z (nm)", 250);
        cmdParser.addParameterFloat("-na" ,"Numerical aperture", 1.4);
        cmdParser.addParameterFloat("-ni" ,"Refractive index immersion", 1.5);
        cmdParser.addParameterFloat("-ns" ,"Refractive index sample", 1.3);
        cmdParser.addParameterFloat("-ti" ,"Working distance (mum)", 150);
        
        cmdParser.setMan("Generate a Gibson Lanni PSF");
        cmdParser.parse(4);

        unsigned int sx = cmdParser.getParameterInt("-width");
        unsigned int sy = cmdParser.getParameterInt("-height");
        unsigned int sz = cmdParser.getParameterInt("-depth");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        float lambda = cmdParser.getParameterFloat("-wavelength");
        float res_lateral = cmdParser.getParameterFloat("-psxy");
        float res_axial = cmdParser.getParameterFloat("-psz");
        float numerical_aperture = cmdParser.getParameterFloat("-na");
        float ni = cmdParser.getParameterFloat("-ni");
        float ns = cmdParser.getParameterFloat("-ns");
        float ti0 = cmdParser.getParameterFloat("-ti");

        if (sx < 4 || sy < 4 || sz < 3){
            observer->message("width must be >= 4, height must be >= 4 and depth must be >= 3 ");
            return 1;
        }

        if (outputImageFile == ""){
            observer->message("Output image file path is empty");
            return 1;
        }

        float* buffer_out = new float[sx*sy*sz];
        float particle_axial_position = 0;
        float ng = 1.5;
        float ng0 = 1.5;
        float ni0 = ni;
        SImg::gibson_lanni_psf(buffer_out, sx, sy, sz, 
                         res_lateral, res_axial, numerical_aperture, lambda,
                         ti0, ni0, ni, ng0, ng, ns, particle_axial_position);

        SImageFloat* image = new SImageFloat(buffer_out, sx, sy, sz);
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
