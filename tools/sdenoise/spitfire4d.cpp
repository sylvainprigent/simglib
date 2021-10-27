#include <score>
#include <scli>
#include <simageio>
#include <sdenoise>
#include <smanipulate>
#include <spadding>


int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        // Parse inputs
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file (txt)");
        cmdParser.addOutputData("-o", "Output image file (txt)");

        cmdParser.addParameterSelect("-method", "Deconvolution method 'SV' or 'HV", "HV");
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 2);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterFloat("-deltaz", "Delta resolution between xy and z", 1.0);
        cmdParser.addParameterFloat("-deltat", "Delta resolution int t", 1.0);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", false);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Denoise a 3D+t image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const float deltaz = cmdParser.getParameterFloat("-deltaz");
        const float deltat = cmdParser.getParameterFloat("-deltat");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool padding = cmdParser.getParameterBool("-padding");
        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("spitfire4d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("spitfire4d: input image: " + inputImageFile);
            observer->message("spitfire4d: output image: " + outputImageFile);
            observer->message("spitfire4d: method: " + method);
            observer->message("spitfire4d: regularization parameter: " + std::to_string(regularization));
            observer->message("spitfire4d: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire4d: nb iterations: " + std::to_string(niter));
        }

        // Run process

        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        unsigned int st = inputImage->getSizeT();
        if (inputImage->getSizeC() > 1)
        {
            throw SException("spitfire4d can process only 3D+t gray scale images");
        }
        float imax = inputImage->getMax();
        float imin = inputImage->getMin();

        SObservable * observable = new SObservable();
        observable->addObserver(observer);

        if (padding)
        {
            if (verbose){
                observer->message("spitfire4d: use padding");
            }
            // padding 
            unsigned int sx_pad = sx + 12;
            unsigned int sy_pad = sy + 12;
            unsigned int sz_pad = sz + 6;
            unsigned int st_pad = st;
            float* noisy_padded_image = new float[sx_pad*sy_pad*sz_pad*st_pad];
            SImg::mirror_padding_4d(noisy_image, noisy_padded_image, sx, sy, sz, st, sx_pad, sy_pad, sz_pad); 

            // min max normalize intensities
            float* noisy_image_norm = new float[sx_pad*sy_pad*sz_pad*st_pad];
            SImg::normMinMax(noisy_padded_image, sx_pad, sy_pad, sz_pad, st_pad, 1, noisy_image_norm);
            delete inputImage;

            float* denoised_image = new float[sx_pad*sy_pad*sz_pad*st_pad];
            SImg::tic();
            if (method == "SV"){
                SImg::spitfire4d_sv(noisy_image_norm, sx_pad, sy_pad, sz_pad, st_pad, denoised_image, regularization, weighting, niter, deltaz, deltat, verbose, observable);
            }
            else if (method == "HV")
            {
                SImg::spitfire4d_hv(noisy_image_norm, sx_pad, sy_pad, sz_pad, st_pad, denoised_image, regularization, weighting, niter, deltaz, deltat, verbose, observable);
            }
            else{
                throw SException("spitfire4d: method must be SV or HV");
            }
            SImg::toc();

            // normalize back intensities
            float omin = denoised_image[0];
            float omax = denoised_image[0];
            for (unsigned int i = 1; i < sx_pad*sy_pad*sz_pad*st_pad; ++i)
            {
                float val = denoised_image[i];
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
            for (unsigned int i = 0; i < sx_pad*sy_pad*sz_pad*st_pad; ++i)
            {
                denoised_image[i] = (denoised_image[i] - omin)/(omax-omin);
                denoised_image[i] = denoised_image[i] * (imax - imin) + imin;
            }

            // remove padding
            float* output = new float[sx*sy*sz*st];
            SImg::remove_padding_4d(denoised_image, output, sx_pad, sy_pad, sz_pad, st_pad, sx, sy, sz);
            delete[] denoised_image;

            SImageReader::write(new SImageFloat(output, sx, sy, sz, st), outputImageFile);

            delete[] noisy_image_norm;    
        }
        else
        {
            // min max normalize intensities
            float* noisy_image_norm = new float[sx*sy*sz*st];
            SImg::normMinMax(noisy_image, sx, sy, sz, st, 1, noisy_image_norm);
            delete inputImage;

            float* denoised_image = new float[sx*sy*sz*st];
            SImg::tic();
            if (method == "SV"){
                SImg::spitfire4d_sv(noisy_image_norm, sx, sy, sz, st, denoised_image, regularization, weighting, niter, deltaz, deltat, verbose, observable);
            }
            else if (method == "HV")
            {
                SImg::spitfire4d_hv(noisy_image_norm, sx, sy, sz, st, denoised_image, regularization, weighting, niter, deltaz, deltat, verbose, observable);
            }
            else{
                throw SException("spitfire4d: method must be SV or HV");
            }
            SImg::toc();

            // normalize back intensities
            float omin = denoised_image[0];
            float omax = denoised_image[0];
            for (unsigned int i = 1; i < sx*sy*sz*st; ++i)
            {
                float val = denoised_image[i];
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
            for (unsigned int i = 0; i < sx*sy*sz*st; ++i)
            {
                denoised_image[i] = (denoised_image[i] - omin)/(omax-omin);
                denoised_image[i] = denoised_image[i] * (imax - imin) + imin;
            }

            SImageReader::write(new SImageFloat(denoised_image, sx, sy, sz, st), outputImageFile);

            delete[] noisy_image_norm;
            delete[] denoised_image;
        }
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
