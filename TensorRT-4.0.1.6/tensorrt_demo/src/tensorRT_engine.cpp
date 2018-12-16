#include <tensorRT_engine.h>
#include <util/blockingQueue.h>
#include <thread>
#include <iostream>
#include <chrono>

#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <vector>
#include <map>
#include <random>
#include <iterator>

#include <parameter.h>

#define CUDACHECK(status)                            \
    {                                                \
        if (status != 0)                             \
        {                                            \
            std::cout << "Cuda failure: " << status; \
            abort();                                 \
        }                                            \
    }

static inline int volume(Dims3 dims)
{
    return dims.d[0] * dims.d[1] * dims.d[2];
}

static std::vector<std::string> gInputs;
static std::map<std::string, Dims3> gInputDimensions;

static float percentile(float percentage, std::vector<float> &times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude < all)
    {
        std::sort(times.begin(), times.end());
        return times[all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

// Logger for TensorRT info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO || gParams.verbose)
            std::cout << msg << std::endl;
    }
};

static Logger gLogger;

class RndInt8Calibrator : public IInt8EntropyCalibrator
{
  public:
    RndInt8Calibrator(int totalSamples, std::string cacheFile)
        : mTotalSamples(totalSamples), mCurrentSample(0), mCacheFile(cacheFile)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        for (auto &elem : gInputDimensions)
        {
            int elemCount = volume(elem.second);

            std::vector<float> rnd_data(elemCount);
            for (auto &val : rnd_data)
                val = distribution(generator);

            void *data;
            CUDACHECK(cudaMalloc(&data, elemCount * sizeof(float)));
            CUDACHECK(cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float), cudaMemcpyHostToDevice));

            mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
        }
    }

    ~RndInt8Calibrator()
    {
        for (auto &elem : mInputDeviceBuffers)
            CUDACHECK(cudaFree(elem.second));
    }

    int getBatchSize() const override
    {
        return 1;
    }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) override
    {
        if (mCurrentSample >= mTotalSamples)
            return false;

        for (int i = 0; i < nbBindings; ++i)
            bindings[i] = mInputDeviceBuffers[names[i]];

        ++mCurrentSample;
        return true;
    }

    const void *readCalibrationCache(size_t &length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void *cache, size_t length) override
    {
    }

  private:
    int mTotalSamples;
    int mCurrentSample;
    std::string mCacheFile;
    std::map<std::string, void *> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
};

static ICudaEngine *caffeToTRTModel()
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();
    const IBlobNameToTensor *blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
                                                              gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(),
                                                              *network,
                                                              gParams.fp16 ? DataType::kHALF : DataType::kFLOAT);

    if (!blobNameToTensor)
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3 &&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
        std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    // specify which tensors are outputs
    for (auto &s : gParams.outputs)
    {
        if (blobNameToTensor->find(s.c_str()) == nullptr)
        {
            std::cout << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3 &&>(network->getOutput(i)->getDimensions());
        std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
                  << dims.d[2] << std::endl;
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(size_t(gParams.workspaceSize) << 20);
    builder->setFp16Mode(gParams.fp16);

    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    if (gParams.int8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(&calibrator);
    }

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

static ICudaEngine *uffToTRTModel()
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    IUffParser *parser = createUffParser();

    // specify which tensors are outputs
    for (auto &s : gParams.outputs)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            std::cerr << "Failed to register output " << s << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    for (auto &s : gParams.uffInputs)
    {
        if (!parser->registerInput(s.first.c_str(), s.second, UffInputOrder::kNCHW))
        {
            std::cerr << "Failed to register input " << s.first << std::endl;
            return nullptr;
        }
    }

    if (!parser->parse(gParams.uffFile.c_str(), *network, gParams.fp16 ? DataType::kHALF : DataType::kFLOAT))
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3 &&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(gParams.workspaceSize << 20);
    builder->setFp16Mode(gParams.fp16);

    RndInt8Calibrator calibrator(1, gParams.calibrationCache);
    if (gParams.int8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(&calibrator);
    }

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cout << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

ICudaEngine *onnxToTRTModel()
{
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // create onnx config file
    nvonnxparser::IOnnxConfig *config = nvonnxparser::createONNXConfig();
    config->setModelFileName(gParams.onnxModelFile.c_str());

    // parse the onnx model to populate the network, then set the outputs
    nvonnxparser::IONNXParser *parser = nvonnxparser::createONNXParser(*config);

    if (!parser->parse(gParams.onnxModelFile.c_str(), DataType::kFLOAT))
    {
        std::cout << "failed to parse onnx file" << std::endl;
        return nullptr;
    }

    // Retrieve the network definition from parser
    if (!parser->convertToTRTNetwork())
    {
        std::cout << "failed to convert onnx network into TRT network" << std::endl;
        return nullptr;
    }

    nvinfer1::INetworkDefinition *network = parser->getTRTNetwork();

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3 &&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        gParams.outputs.push_back(network->getOutput(i)->getName());
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(gParams.workspaceSize << 20);
    builder->setFp16Mode(gParams.fp16);

    ICudaEngine *engine = builder->buildCudaEngine(*network);

    if (engine == nullptr)
    {
        std::cout << "could not build engine" << std::endl;
        assert(false);
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

void createMemory(const ICudaEngine &engine, std::vector<void *> &buffers, const std::string &name)
{
    size_t bindingIndex = engine.getBindingIndex(name.c_str());
    printf("name=%s, bindingIndex=%d, buffers.size()=%d\n", name.c_str(), (int)bindingIndex, (int)buffers.size());
    assert(bindingIndex < buffers.size());
    Dims3 dimensions = static_cast<Dims3 &&>(engine.getBindingDimensions((int)bindingIndex));
    size_t eltCount = dimensions.d[0] * dimensions.d[1] * dimensions.d[2] * gParams.batchSize, memSize = eltCount * sizeof(float);

    float *localMem = new float[eltCount];
    for (size_t i = 0; i < eltCount; i++)
        localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

    void *deviceMem;
    CUDACHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    CUDACHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

    delete[] localMem;
    buffers[bindingIndex] = deviceMem;
}

void doInference(ICudaEngine &engine)
{
    IExecutionContext *context = engine.createExecutionContext();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.

    std::vector<void *> buffers(gInputs.size() + gParams.outputs.size());
    for (size_t i = 0; i < gInputs.size(); i++)
        createMemory(engine, buffers, gInputs[i]);

    for (size_t i = 0; i < gParams.outputs.size(); i++)
        createMemory(engine, buffers, gParams.outputs[i]);

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CUDACHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CUDACHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    std::vector<float> times(gParams.avgRuns);
    for (int j = 0; j < gParams.iterations; j++)
    {
        float total = 0, ms;
        for (int i = 0; i < gParams.avgRuns; i++)
        {
            if (gParams.hostTime)
            {
                auto tStart = std::chrono::high_resolution_clock::now();
                context->execute(gParams.batchSize, &buffers[0]);
                auto tEnd = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            }
            else
            {
                cudaEventRecord(start, stream);
                context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
                cudaEventRecord(end, stream);
                cudaEventSynchronize(end);
                cudaEventElapsedTime(&ms, start, end);
            }
            times[i] = ms;
            total += ms;
        }
        total /= gParams.avgRuns;
        std::cout << "Average over " << gParams.avgRuns << " runs is " << total << " ms (percentile time is " << percentile(gParams.pct, times) << ")." << std::endl;
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();
}

static ICudaEngine *createEngine()
{
    ICudaEngine *engine;
    if ((!gParams.deployFile.empty()) || (!gParams.uffFile.empty()) || (!gParams.onnxModelFile.empty()))
    {

        if (!gParams.uffFile.empty())
        {
            engine = uffToTRTModel();
        }
        else if (!gParams.onnxModelFile.empty())
        {
            engine = onnxToTRTModel();
        }
        else
        {
            engine = caffeToTRTModel();
        }

        if (!engine)
        {
            std::cerr << "Engine could not be created" << std::endl;
            return nullptr;
        }

        if (!gParams.engine.empty())
        {
            std::ofstream p(gParams.engine);
            if (!p)
            {
                std::cerr << "could not open plan output file" << std::endl;
                return nullptr;
            }
            IHostMemory *ptr = engine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char *>(ptr->data()), ptr->size());
            ptr->destroy();
        }
        return engine;
    }

    // load directly from serialized engine file if deploy not specified
    if (!gParams.engine.empty())
    {
        char *trtModelStream{nullptr};
        size_t size{0};
        std::ifstream file(gParams.engine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

        IRuntime *infer = createInferRuntime(gLogger);
        engine = infer->deserializeCudaEngine(trtModelStream, size, nullptr);
        if (trtModelStream)
            delete[] trtModelStream;

        // assume input to be "data" for deserialized engine
        gInputs.push_back("data");
        return engine;
    }

    // complain about empty deploy file
    std::cerr << "Deploy file not specified" << std::endl;
    return nullptr;
}

template <typename Ftype, typename Btype>
class GPUDevice
{
  public:
    GPUDevice(size_t device_id, size_t batch_size)
    {
        this->device_id_ = device_id;
        this->batch_size_ = batch_size;
        LOG(INFO) << "create device: " << device_id_;
    }
    ~GPUDevice()
    {
        if (tp_ != nullptr)
            tp_->join();
        std::cout << "GPU device exit..." << std::endl;
    }

    void setup(BlockingQueue<Btype> *front[2], BlockingQueue<Btype> *back[2])
    {

        for (int index = 0; index < 2; index++)
        {
            this->front_[index] = front[index];
            this->back_[index] = back[index];
        }
        tp_ = new std::thread(&GPUDevice::main_loop, this);
    }

    void main_loop()
    {
        {
            cudaSetDevice(device_id_);
            ICudaEngine *engine = createEngine();
            if (!engine)
            {
                std::cerr << "Engine could not be created" << std::endl;
                exit(-1);
            }

            if (gParams.uffFile.empty() && gParams.onnxModelFile.empty())
            {
                nvcaffeparser1::shutdownProtobufLibrary();
                std::cout << "NVCaffe parser shutdown..." << std::endl;
            }
            else if (gParams.deployFile.empty() && gParams.onnxModelFile.empty())
            {
                nvuffparser::shutdownProtobufLibrary();
                std::cout << "nvuffparser parser shutdown..." << std::endl;
            }
            doInference(*engine);
            engine->destroy();
        }
        LOG(INFO) << "run inference ... on device: " << device_id_;
        while (1)
        {
            /*
                std::this_thread::sleep_for(std::chrono::seconds(10));
                LOG(INFO) << "run inference ... on device: " << device_id_;
            */
            front_[1]->push(front_[0]->pop());
        }
    }

  private:
    std::thread *tp_;
    size_t device_id_;
    size_t batch_size_;
    BlockingQueue<Btype> *front_[2];
    BlockingQueue<Btype> *back_[2];
};

TensorRTEngine::TensorRTEngine(std::string nm)
{

    this->name_ = nm;
    LOG(INFO) << " Create TensorRT engine " << nm;
}

TensorRTEngine::~TensorRTEngine()
{
    LOG(INFO) << name_ << " TensorRT engine release";
}

void TensorRTEngine::setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2])
{
    LOG(INFO) << name_ << " Engine setup";
    CHECK(gParams.initilized) << "engine setup failed....., you must init parameter first";
    std::vector<GPUDevice<int, int> *> egInstance;
    for (size_t index = 0; index < 2; index++)
        egInstance.push_back(new GPUDevice<int, int>(index, 32));
    LOG(INFO) << "launch engine for 5 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    for (auto &ieg : egInstance)
    {
        ieg->setup(from, to);
    }
}

std::string TensorRTEngine::display()
{
    LOG(INFO) << "This engine brand is: " << name_;
    return name_;
}