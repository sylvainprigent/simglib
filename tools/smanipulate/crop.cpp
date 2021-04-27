#include <iostream>

#include <smanipulate>
#include <simageio>
#include <scli>

int main(int argc, char *argv[])
{
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterInt("-x1", "X coordinate of the top left corner", 12);
        cmdParser.addParameterInt("-y1", "Y coordinate of the top left corner", 12);
        cmdParser.addParameterInt("-x2", "X coordinate of the bottom right corner", 24);
        cmdParser.addParameterInt("-y2", "Y coordinate of the bottom right corner", 24);

        cmdParser.parse(2);


        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const int x1 = cmdParser.getParameterInt("-x1");
        const int y1 = cmdParser.getParameterInt("-y1");
        const int x2 = cmdParser.getParameterInt("-x2");
        const int y2 = cmdParser.getParameterInt("-y2");

        if (inputImageFile == ""){
            std::cout << "crop: Input image path is empty" << std::endl;
            return 1;
        }

        SImageFloat* image = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile));

        float* image_buffer =  image->getBuffer();
        unsigned int sx = image->getSizeX();
        unsigned int sy = image->getSizeY();
        unsigned int sz = image->getSizeZ();
        unsigned int st = image->getSizeT();
        unsigned int sc = image->getSizeC();  
        float* output_buffer;

        SImg::crop(image_buffer, sx, sy, sz, st, sc, output_buffer, x1, x2, y1, y2, -1, -1, -1, -1, -1, -1);
        SImageFloat* out_image = new SImageFloat(output_buffer, sx, sy, sz, st, sc);

        SImageReader::write(out_image, outputImageFile);

        // delete
        delete image;
        delete out_image;
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
