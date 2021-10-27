#include <score>
#include <scli>
#include <simageio>
#include <smanipulate>
#include <sdeconv>
#include <sfiltering>
#include <sfft>
#include <sdata>
#include <sdataio>
#include <spadding>
#include "math.h"

using namespace SImg;

void normalize_back_intensities_(float* deconv_image, int bs, float imin, float imax)
{
    // normalize back intensities
    float omin = deconv_image[0];
    float omax = deconv_image[0];
    for (unsigned int i = 1; i < bs; ++i)
    {
        float val = deconv_image[i];
        if (val > omax)
        {
            omax = val;
       }
        if (val < omin)
        {
            omin = val;
        }
    }

#pragma omp parallel for
    for (unsigned int i = 0; i < bs; ++i)
    {
        deconv_image[i] = (deconv_image[i] - omin)/(omax-omin);
        deconv_image[i] = deconv_image[i] * (imax - imin) + imin;
    }
}

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        // Parse inputs
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image directory");

        cmdParser.addParameterFloat("-sigma", "PSF sigma (gaussian)", 1.5);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Estimate the regularisation parameter for 2D HV SPITFIRe");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageDir = cmdParser.getDataURI("-o");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("spitfire2d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("spitfire2d: input image: " + inputImageFile);
            observer->message("spitfire2d: output dir: " + outputImageDir);
            observer->message("spitfire2d: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire2d: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* blurry_image_i = inputImage->getBuffer();
        unsigned int sx_i = inputImage->getSizeX();
        unsigned int sy_i = inputImage->getSizeY();
        if (inputImage->getSizeZ() > 1 || inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("spitfire2d can process only 2D gray scale images");
        }

        // add padding to the input image
        unsigned int pad = 14;
        unsigned int sx = sx_i + 2*pad;
        unsigned int sy = sy_i + 2*pad;    
        float* blurry_image = new float[sx*sy];

        int ctrl = hanning_padding_2d(blurry_image_i, blurry_image, sx_i, sy_i, sx, sy);
        //int ctrl = mirror_padding_2d(blurry_image_i, blurry_image, sx_i, sy_i, sx, sy); 
        //int ctrl = padding_2d(blurry_image_i, blurry_image, sx_i, sy_i, sx, sy);
        SImageReader::write(new SImageFloat(blurry_image, sx, sy), "mirror_hanning.tif");
        if (ctrl > 0){
            observer->message("ZeroPadding: dimensions missmatch", SObserver::MessageTypeError);
            return 1;
        }

        // create the PSF
        float* psf = new float[sx*sy];
        SImg::gaussian_psf_2d(psf, sx, sy, sigma, sigma);
        float psf_sum = 0.0;
        for (unsigned int i = 0 ; i < sx*sy ; ++i){
            psf_sum += psf[i]; 
        }
        for (unsigned int i = 0 ; i < sx*sy ; ++i){
            psf[i] /= psf_sum;
        }

        // normalize the input image
        unsigned int bs = sx * sy;
        float imin = blurry_image[0];
        float imax = blurry_image[0];
        for (unsigned int i = 1; i < bs; ++i)
        {
            float val = blurry_image[i];
            if (val > imax)
            {
                imax = val;
            }
            if (val < imin)
            {
                imin = val;
            }
        }

        float *blurry_image_norm = new float[sx * sy];
        normL2(blurry_image, sx, sy, 1, 1, 1, blurry_image_norm);
        //delete blurry_image;

        // command line observer
        SObservable* observable = new SObservable();
        observable->addObserver(observer);    

        float* conv_output = new float[sx*sy];
        conv_output = convolution_2d(blurry_image_norm, psf, sx, sy);
        SImageReader::write(new SImageFloat(conv_output, sx, sy), "conv_test.tif");

        // try several regularizations
        std::vector<float> lambdas;
        std::vector<float> energies;
        std::vector<float> energies_before;
        std::vector<float> tvs;

        //for (float lambda = 0.998; lambda <= 1; lambda+=0.00001)
        for (int pow_lambda = 1; pow_lambda <= 30; pow_lambda++)
        {
            float lambdas_prim = pow(2.0, -pow_lambda);
            float lambda = 1/(1+lambdas_prim);    

            std::cout << "pow_lambda = " << pow_lambda << std::endl;
            std::cout << "lambdas_prim = " << lambdas_prim << std::endl;
            std::cout << "lambda = " << lambda << std::endl;

            //float lambdas_prim = (1-lambda)/lambda; 
            float *deconv_image = (float *)malloc(sizeof(float) * (sx*sy));
            spitfire2d_deconv_hv(blurry_image_norm, sx, sy, psf, deconv_image, lambdas_prim, weighting, niter, verbose, observable);
            std::string outputImageFile = outputImageDir + "/lambda_" + std::to_string(pow_lambda) + ".tif";
            
            float energy_before = energy_hv(blurry_image_norm, sx, sy, psf, deconv_image, lambdas_prim, weighting);
            normalize_back_intensities_(deconv_image, bs, imin, imax); 
            float energy = energy_hv(blurry_image_norm, sx, sy, psf, deconv_image, lambdas_prim, weighting);

            float tv = gradient2dL1(deconv_image, sx, sy, 1, 1, 1);

            observer->message("energy = " + std::to_string(energy));
            observer->message("energy before = " + std::to_string(energy_before));
            observer->message("tv = " + std::to_string(tv));

               
            float *deconv_image_nopad =  new float[sx_i*sy_i];   
            remove_padding_2d(deconv_image, deconv_image_nopad, sx, sy, sx_i, sy_i);
            SImageReader::write(new SImageFloat(deconv_image_nopad, sx_i, sy_i), outputImageFile);
            delete deconv_image;
            delete[] deconv_image_nopad;

            lambdas.push_back(pow_lambda);
            energies.push_back(energy);
            energies_before.push_back(energy_before);
            tvs.push_back(tv);
        }

        STable* table = new STable(lambdas.size(), 4);
        table->setHeader(0, "lambda");
        table->setHeader(1, "energy_before");
        table->setHeader(2, "energy");
        table->setHeader(3, "tv");
        for (int i = 0 ; i < lambdas.size() ; i++)
        {
            table->set(i, 0, lambdas[i]);
            table->set(i, 1, energies_before[i]);
            table->set(i, 2, energies[i]);
            table->set(i, 3, tvs[i]);
        }  

        SCSV scsv;
        scsv.set(table);
        scsv.write(outputImageDir + "/energies.csv");

        delete[] blurry_image_norm;
        delete observable;
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
