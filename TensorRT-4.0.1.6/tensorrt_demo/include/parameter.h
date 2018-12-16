#ifndef __PARAMETER_LAUNCH_H__
#define __PARAMETER_LAUNCH_H__
#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvUffParser.h"
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace nvonnxparser;

struct Params
{
    std::string deployFile, modelFile, engine, calibrationCache{"CalibrationTable"};
    std::string uffFile;
    std::string onnxModelFile;
    std::vector<std::string> outputs;
    std::vector<std::pair<std::string, Dims3>> uffInputs;
    int device{0}, batchSize{1}, workspaceSize{16}, iterations{10}, avgRuns{10};
    bool fp16{false}, int8{false}, verbose{false}, hostTime{false};
    float pct{99};
    bool initilized{false};
};
extern Params gParams;

bool parseArgs(int argc, const char *argv[]);
#endif