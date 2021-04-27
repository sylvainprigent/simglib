#include <iostream>
#include <score>
#include <scli>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input file");
        cmdParser.addOutputData("-o", "Output file");

        cmdParser.addParameterFloat("-f", "Parameter float ", 2);
        cmdParser.addParameterString("-s", "Parameter string", "Hello ");
        cmdParser.setMan("Demo of cmp parser");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float param_float = cmdParser.getParameterFloat("-f");
        const std::string param_string = cmdParser.getParameterString("-s");

        if (inputImageFile == ""){
            observer->message("Demo needs an input file");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("Demo: input file: " + inputImageFile);
            observer->message("Demo: output file: " + outputImageFile);
            observer->message("Demo: parameter float: " + std::to_string(param_float));
            observer->message("Demo: parameter string: " + param_string);
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
