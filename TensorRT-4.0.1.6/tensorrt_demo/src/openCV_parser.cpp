#include <openCV_parser.h>
#include <util/blockingQueue.h>
#include <thread>
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <util/mylogs.h>
#include <fstream>
#include <cuda_runtime_api.h>

#define CUDACHECK(status)                            \
    {                                                \
        if (status != 0)                             \
        {                                            \
            std::cout << "Cuda failure: " << status; \
            abort();                                 \
        }                                            \
    }

#include <sys/time.h>
static uint64_t current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

class DecoderHandler
{
  public:
    DecoderHandler(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2])
    {
        for (int index = 0; index < 2; index++)
        {
            this->channel_toGPU[index] = to[index];
            this->channel_fromCollector[index] = from[index];
        }
    }
    void setup(size_t device_id, size_t batch_size)
    {
        this->device_id_ = device_id;
        this->batch_size_ = batch_size;
        LOG(INFO) << "create device: " << device_id_;
        tp_ = new std::thread(&DecoderHandler::main_loop, this);
    }
    ~DecoderHandler()
    {
        if (tp_ != nullptr)
            tp_->join();
        std::cout << "GPU device exit..." << std::endl;
    }

    void main_loop()
    {
        BlockingQueue<int> *bq = new BlockingQueue<int>();
        std::vector<BlockingQueue<int> *> b_vec;

        uint32_t tmp_image_size = 3 * 227 * 227;
        uint32_t tmp_batch_size = 32;

        cudaStream_t copystream[2]; //create two copy stream...
        CUDACHECK(cudaSetDevice(device_id_));

        CUDACHECK(cudaStreamCreate(&copystream[0]));
        CUDACHECK(cudaStreamCreate(&copystream[1]));

        void *placeholder[16];

        for (int index = 0; index < 16; index++)
        {
            CUDACHECK(cudaMalloc((void **)&(placeholder[index]), tmp_image_size * tmp_batch_size));
        }

        char *hostmem = nullptr;
        CUDACHECK(cudaHostAlloc((void **)&hostmem, tmp_image_size, cudaHostAllocDefault));

        //creative concrete parser, e.g. opencv, nvjpeg or aipre
        for (int index = 0; index < 10; index++)
        {
            slaved_decoder.push_back(new std::thread(&DecoderHandler::decoder, this, index, bq));
        }
        long long ii = 0;
        long long cc = 0;
        uint64_t before = current_time();
        uint64_t after = 0;
        std::queue<void *> in_flight;
        size_t accumulated_num = 0;
        int stream_index = 0;
        int left = 0;
        bool fetched = true;
        while (1)
        {
            std::vector<int> retrive;
            bq->pop_all(retrive);

            accumulated_num %= 32;
            left += retrive.size();
            cc += retrive.size();
            if (cc >= 1000 * 50)
            {
                after = current_time();
                LOG(INFO) << "Parser serve GPU " << device_id_ << " from " << bq->pop("Get from reader...") << std::endl
                          << "Parser: Rate: " << cc / ((after - before) / 1000.0 / 1000.0) << " images/s" << std::endl
                          << "time: " << (after - before) << " num: " << cc;
                cc = 0;
                before = after;
            }
            while (left > 0)
            {

                CUDACHECK(cudaMemcpyAsync(static_cast<char *>(placeholder[stream_index]) + accumulated_num * tmp_image_size, hostmem, tmp_image_size, cudaMemcpyHostToDevice, copystream[stream_index]));
                accumulated_num++, left--;
                accumulated_num %= 32;
                if (accumulated_num == 0)
                {
                    if (cudaStreamQuery(copystream[(stream_index + 2 - 1) % 2]) != cudaSuccess)
                    {
                        LOG(INFO) << "last stream was not finished yet...";
                        //todo, add to upper queue.

                        //sync last stream...
                        CUDACHECK(cudaStreamSynchronize(copystream[(stream_index + 2 - 1) % 2]));
                        channel_toGPU[0]->push(stream_index);
                    }
                    stream_index++;
                    stream_index %= 2;
                    fetched = false;
                    break;
                }
            }
            if (!fetched)
            {
                if (cudaStreamQuery(copystream[(stream_index + 2 - 1) % 2]) == cudaSuccess) //all work has been executed
                {
                    //todo, add to upper queue.
                    channel_toGPU[0]->push(stream_index);
                    /*
                    LOG(INFO) << "Size:: " << channel_toGPU[0]->size();
                    */
                }
            }

            if (ii++ % (1000 * 50) == 0)
            {
                LOG(INFO) << "Parser serve GPU " << device_id_ << " from " << bq->pop("Get from reader...") << std::endl
                          << "Get size: " << bq->size();
            }
        }
    }

    void decoder(int index, BlockingQueue<int> *bq_)
    {
        BlockingQueue<int> **from = channel_fromCollector;
        char tmp[1024 * 300] = {0};
        size_t file_size = 0;
        {
            std::ifstream file;
            file.open("/home/newplan/software/tensorrt/TensorRT-4.0.1.6/tensorrt_demo/data/runhuzhang.jpg", std::ios::binary);
            if (!file)
            {
                std::cout << "open file failed..." << std::endl;
            }
            file.seekg(0, std::ios::end);
            file_size = file.tellg();
            std::cout << "File size: " << file_size << " Bytes." << std::endl;
            file.seekg(0, std::ios::beg);
            file.read(tmp, sizeof(tmp));
            file.close();
            if (file_size >= 1024 * 300)
            {
                LOG(INFO) << "error in file malloc...." << std::endl;
            }
        }

        int inddd = 0;
        uint64_t before = current_time();
        while (1)
        {
            inddd++;
            /*
            DLOG(INFO) << "decoder " << index << "... to serve GPU " << device_id_;
            */
            //std::this_thread::sleep_for(std::chrono::seconds(20));

            if (1)
            {
                from[0]->pop("parser was blocked by data collector");
                std::vector<char> data(tmp, tmp + file_size);
                cv::Mat src = cv::imdecode(cv::Mat(data), CV_LOAD_IMAGE_COLOR);
                cv::Mat dst;
                cv::resize(src, dst, cv::Size(227, 227), 0, 0, cv::INTER_LINEAR);

                if (inddd % 10000 == 0)
                {
                    uint64_t after = current_time();
                    LOG_IF(INFO, index == 0) << "each resize processing time is: " << (after - before) / 1000.0 / inddd
                                             << " ms" << std::endl
                                             << "Rate: " << inddd * 1.0 / ((after - before) / 1000.0 / 1000) << " image/s/CPU";
                    before = after;
                    inddd = 0;
                }
                bq_->push(index);
                from[1]->push(0);
            }
        }
        cv::waitKey(0);
    }

  private:
    std::thread *tp_;
    size_t device_id_;
    size_t batch_size_;
    BlockingQueue<int> *channel_toGPU[2];
    BlockingQueue<int> *channel_fromCollector[2];
    BlockingQueue<int> *subscripter;
    std::vector<std::thread *> slaved_decoder;
};

OpenCVParser::OpenCVParser(std::string nm)
{
    this->name_ = nm;
    LOG(INFO) << " Create OpenCVParser " << nm;
}

OpenCVParser::~OpenCVParser()
{
    LOG(INFO) << name_ << " OpenCVParser release";
}

void OpenCVParser::setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2])
{
    LOG(INFO) << name_ << "Parser setup";

    for (size_t index = 0; index < 2; index++)
    {
        hder.push_back(new DecoderHandler(from, to));
    }
    for (size_t index = 0; index < 2; index++)
    {
        hder[index]->setup(index, 32);
    }
}

std::string OpenCVParser::display()
{
    LOG(INFO) << "This OpenCVParser brand is: " << name_;
    return name_;
}