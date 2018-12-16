#include <parameter.h>
#include <iostream>
#include <sstream>

#include <cstring>

Params gParams;
static void printUsage()
{
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --deploy=<file>      Caffe deploy file\n");
    printf("  OR --uff=<file>      UFF file\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

    printf("\nMandatory params for onnx:\n");
    printf("  --onnx=<file>        ONNX Model file\n");

    printf("\nOptional params:\n");

    printf("  --uffInput=<name>,C,H,W Input blob names along with their dimensions for UFF parser\n");
    printf("  --model=<file>       Caffe model file (default = no model, random weights used)\n");
    printf("  --batch=N            Set batch size (default = %d)\n", gParams.batchSize);
    printf("  --device=N           Set cuda device to N (default = %d)\n", gParams.device);
    printf("  --iterations=N       Run N iterations (default = %d)\n", gParams.iterations);
    printf("  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
    printf("  --percentile=P       For each iteration, report the percentile time at P percentage (0<P<=100, default = %.1f%%)\n", gParams.pct);
    printf("  --workspace=N        Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
    printf("  --fp16               Run in fp16 mode (default = false). Permits 16-bit kernels\n");
    printf("  --int8               Run in int8 mode (default = false). Currently no support for ONNX model.\n");
    printf("  --verbose            Use verbose logging (default = false)\n");
    printf("  --hostTime           Measure host time rather than GPU time (default = false)\n");
    printf("  --engine=<file>      Generate a serialized TensorRT engine\n");
    printf("  --calib=<file>       Read INT8 calibration cache file.  Currently no support for ONNX model.\n");

    fflush(stdout);
}
static bool parseString(const char *arg, const char *name, std::string &value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

static std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

static bool parseInt(const char *arg, const char *name, int &value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

static bool parseBool(const char *arg, const char *name, bool &value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
    if (match)
    {
        std::cout << name << std::endl;
        value = true;
    }
    return match;
}

static bool parseFloat(const char *arg, const char *name, float &value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        std::cout << name << ": " << value << std::endl;
    }
    return match;
}

bool parseArgs(int argc, const char *argv[])
{
    if (argc < 2)
    {
        printUsage();
        return false;
    }

    for (int j = 1; j < argc; j++)
    {
        if (parseString(argv[j], "model", gParams.modelFile) || parseString(argv[j], "deploy", gParams.deployFile) || parseString(argv[j], "engine", gParams.engine))
            continue;

        if (parseString(argv[j], "uff", gParams.uffFile))
            continue;

        if (parseString(argv[j], "onnx", gParams.onnxModelFile))
            continue;

        if (parseString(argv[j], "calib", gParams.calibrationCache))
            continue;

        std::string output;
        if (parseString(argv[j], "output", output))
        {
            gParams.outputs.push_back(output);
            continue;
        }

        std::string uffInput;
        if (parseString(argv[j], "uffInput", uffInput))
        {
            std::vector<std::string> uffInputStrs = split(uffInput, ',');
            if (uffInputStrs.size() != 4)
            {
                printf("Invalid uffInput: %s\n", uffInput.c_str());
                return false;
            }

            gParams.uffInputs.push_back(std::make_pair(uffInputStrs[0], Dims3(atoi(uffInputStrs[1].c_str()), atoi(uffInputStrs[2].c_str()), atoi(uffInputStrs[3].c_str()))));
            continue;
        }

        if (parseInt(argv[j], "batch", gParams.batchSize) || parseInt(argv[j], "iterations", gParams.iterations) || parseInt(argv[j], "avgRuns", gParams.avgRuns) || parseInt(argv[j], "device", gParams.device) || parseInt(argv[j], "workspace", gParams.workspaceSize))
            continue;

        if (parseFloat(argv[j], "percentile", gParams.pct))
            continue;

        if (parseBool(argv[j], "fp16", gParams.fp16) || parseBool(argv[j], "int8", gParams.int8) || parseBool(argv[j], "verbose", gParams.verbose) || parseBool(argv[j], "hostTime", gParams.hostTime))
            continue;

        printf("Unknown argument: %s\n", argv[j]);
        return false;
    }
    gParams.initilized = true;
    return true;
}