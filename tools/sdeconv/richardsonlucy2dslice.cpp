#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <spadding>
#include <sdeconv>
#include <smanipulate>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterFloat("-sigma", "Sigma of the PSF", 2);
        cmdParser.addParameterInt("-niter", "Number of iterations", 40);
        cmdParser.addParameterFloat("-lambda", "Regularization parameter", 0);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", true);

        cmdParser.setMan("Deconvolve a 3D image using the Richardson Lucy algorithm slice by slice");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const int niter = cmdParser.getParameterFloat("-niter");
        const float lambda = cmdParser.getParameterFloat("-lambda");
        const bool padding = cmdParser.getParameterBool("-padding");

        if (inputImageFile == ""){
            observer->message("RichardsonLucy2Dslice: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("RichardsonLucy2Dslice: input image: " + inputImageFile);
            observer->message("RichardsonLucy2Dslice: output image: " + outputImageFile);
            observer->message("RichardsonLucy2Dslice: sigma: " + std::to_string(sigma));
            observer->message("RichardsonLucy2Dslice: niter: " + std::to_string(niter));
        }

        // Load input image
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float imin = inputImage->getMin();
        float imax = inputImage->getMax();
        float* buffer_in = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();

        if (padding)
        {
            if (verbose){
                observer->message("RichardsonLucy2Dslice: use padding");
            }
            unsigned int r_pad = 24;
            unsigned int sx_pad = sx + 2*r_pad;
            unsigned int sy_pad = sy + 2*r_pad;

            // create the PSF
            float* psf = new float[sx_pad*sy_pad];
            SImg::gaussian_psf_2d(psf, sx_pad, sy_pad, sigma, sigma);
            SImg::normalize_intensities(psf, sx_pad*sy_pad, "sum");

            // run 2D deconv on each Z slice
            SImageFloat* outputImage = new SImageFloat(sx, sy, sz);
            float * outputImageBuffer = outputImage->getBuffer();
            SImg::tic();
            for (unsigned int z = 0 ; z < inputImage->getSizeZ() ; ++z)
            {
                // get the slice
                SImageFloat* slice = inputImage->getSlice(z);
                float* sliceBuffer = slice->getBuffer();
                float imin = slice->getMin();
                float imax = slice->getMax();

                // padding
                float* blurry_padded_slice = new float[sx_pad*sy_pad];
                SImg::hanning_padding_2d(sliceBuffer, blurry_padded_slice, sx, sy, sx_pad, sy_pad); 
                delete sliceBuffer; 

                // run deconv
                float *outputSliceBuffer = (float *)malloc(sizeof(float) * (sx_pad*sy_pad));
                SImg::normalize_intensities(blurry_padded_slice, sx_pad*sy_pad, "L2");
                if (lambda > 1e-9){
                    SImg::richardsonlucy_tv_2d(blurry_padded_slice, psf, outputSliceBuffer, sx_pad, sy_pad, niter, lambda);
                }
                else{
                    SImg::richardsonlucy_2d(blurry_padded_slice, psf, outputSliceBuffer, sx_pad, sy_pad, niter);
                }
                delete[] blurry_padded_slice;

                SImg::normalize_back_intensities(outputSliceBuffer, sx_pad*sy_pad, 0, imax);

                // copy deblured slice to output image
                for (unsigned int x = 0 ; x < sx ; ++x)
                {
                    for (unsigned int y = 0 ; y < sy ; ++y)
                    {
                        outputImageBuffer[ z + sz*(y + sy*x)] = outputSliceBuffer[y+r_pad + sy_pad*(x+r_pad)];
                    }
                }
                delete outputSliceBuffer;
            }
            SImg::toc();

            SImageReader::write(outputImage, outputImageFile);

            delete outputImage;
            delete inputImage;
        }
        else
        {
            // create the PSF
            float* psf = new float[sx*sy];
            SImg::gaussian_psf_2d(psf, sx, sy, sigma, sigma);
            SImg::normalize_intensities(psf, sx*sy, "sum");

            // run 2D deconv on each Z slice
            SImageFloat* outputImage = new SImageFloat(sx, sy, sz);
            float * outputImageBuffer = outputImage->getBuffer();
            SImg::tic();
            for (unsigned int z = 0 ; z < inputImage->getSizeZ() ; ++z)
            {
                // get the slice
                SImageFloat* slice = inputImage->getSlice(z);
                float* sliceBuffer = slice->getBuffer();
                float imin = slice->getMin();
                float imax = slice->getMax();

                // run deconv
                float *outputSliceBuffer = (float *)malloc(sizeof(float) * (sx*sy));
                SImg::normalize_intensities(sliceBuffer, sx*sy, "L2");
                if (lambda > 1e-9){
                    SImg::richardsonlucy_tv_2d(sliceBuffer, psf, outputSliceBuffer, sx, sy, niter, lambda);
                }
                else{
                    SImg::richardsonlucy_2d(sliceBuffer, psf, outputSliceBuffer, sx, sy, niter);
                }
                SImg::normalize_back_intensities(outputSliceBuffer, sx*sy, imin, imax);

                // copy deblured slice to output image
                for (unsigned int x = 0 ; x < sx ; ++x)
                {
                    for (unsigned int y = 0 ; y < sy ; ++y)
                    {
                        outputImageBuffer[ z + sz*(y + sy*x)] = outputSliceBuffer[y + sy*x];
                    }
                }
                delete outputSliceBuffer;
            }
            SImg::toc();

            SImageReader::write(outputImage, outputImageFile);

            delete outputImage;
            delete inputImage;
        }
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
