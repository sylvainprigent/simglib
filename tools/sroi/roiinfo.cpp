#include <iostream>

#include <sroi>
#include <simageio>
#include <sdataio>
#include <scli>

#include "SRoiInfo.h"

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
        std::string outputCSVFile = cmdParser.getDataURI("-o");

        if (inputImageFile == "" || inputRoiFile == ""){
            std::cout << "roiinfo: need an input image and a ROI file" << std::endl;
            return 1;
        }

        SImageFloat* image = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile));
        SRoiReader roiReader;
        roiReader.setFile(inputRoiFile);
        roiReader.run();
        std::vector<SRoi*> rois = roiReader.getRois();

        std::cout << "image size = " << image->getSizeX() << ", " << image->getSizeY() << std::endl;

        SRoiInfo* process = new SRoiInfo();
        process->setInput(image);
        process->setRois(rois);
        process->run();
        STable* data = process->getOutput();

        SCSV csvWriter;
        csvWriter.set(data);
        csvWriter.write(outputCSVFile);    

        // delete
        delete process;
        delete image;
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
