#include <iostream>

#include <sroi>
#include <simageio>
#include <simage>
#include <sdataio>
#include <scli>

int main(int argc, char *argv[])
{
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addInputData("-rois", "rois file");
        cmdParser.addOutputData("-o", "Output csv file");

        cmdParser.parse(3);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string inputRoiFile = cmdParser.getDataURI("-rois");
        std::string outputFile = cmdParser.getDataURI("-o");

        if (inputImageFile == "" || inputRoiFile == ""){
            std::cout << "roiinfo: need an input image and a ROI file" << std::endl;
            return 1;
        }

        // Run process
        SRoiReader reader;
        reader.setFile(inputRoiFile);
        reader.run();

        SImageFloat* inputImage = SImageCast::toFloat(SImageReader::read(inputImageFile, 32));
        SRoiDrawer drawer;
        drawer.setInput(inputImage);
        drawer.setRois(reader.getRois());
        drawer.run();
        SImage* outImage = drawer.getOutput();

        SImageReader::write(outImage, outputFile);

        delete inputImage;
        delete outImage;   
        return 0;
    }
    catch (SCliException &e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (std::exception &e)
    {
        //sciLogAccess::instance()->log(e.what());
        return 1;
    }

    return 0;

}
