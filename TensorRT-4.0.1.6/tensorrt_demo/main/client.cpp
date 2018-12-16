#include <inference_sys_director.h>
#include <inference_sys_builder.h>
#include <np_inference_sys_builder.h>
#include <np_inference_sys_director.h>
#include <chrono>
#include <thread>
#include <util/blockingQueue.h>
#include <parameter.h>

int main(int argc, char const *argv[])
{
    if (!parseArgs(argc, argv))
    {
        LOG(FATAL) << "error in set up engine";
    }
    log_sys logs(argv[0]);
    InferenceSysDirector *director = new NPInferenceSysDirector(new NPInferenceSysBuilder());
    InferenceSys *infer_sys = director->create_inferenceSys();
    infer_sys->get_engine()->display();
    infer_sys->get_parser()->display();
    infer_sys->get_dataCollector()->display();

    BlockingQueue<int> *collector_2_parser[2];
    BlockingQueue<int> *parser_2_engine[2];
    BlockingQueue<int> *engine_2_collector[2];

    for (int index = 0; index < 2; index++)
    {
        collector_2_parser[index] = new BlockingQueue<int>();
        parser_2_engine[index] = new BlockingQueue<int>();
        engine_2_collector[index] = new BlockingQueue<int>();
    }
    std::string a("100");
    infer_sys->get_engine()->setup(parser_2_engine, engine_2_collector);
    infer_sys->get_parser()->setup(collector_2_parser, parser_2_engine);
    infer_sys->get_dataCollector()->setup(engine_2_collector, collector_2_parser);

    while (1)
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        LOG(INFO) << "main thread loops....";
    }
    return 0;
}
