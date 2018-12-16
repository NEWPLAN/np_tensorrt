#include <np_inference_sys_builder.h>
#include <util/mylogs.h>
#include <tensorRT_engine.h>
#include <openCV_parser.h>
#include <tcp_channel.h>

EngineInterface *NPInferenceSysBuilder::building_engine()
{
  LOG(INFO) << "Building engine....";
  return new TensorRTEngine("TensorRT");
}

DataCollectorInterface *NPInferenceSysBuilder::building_dataCollector()
{

  LOG(INFO) << "Building data collector....";
  return new TCPChannel("TCP");
}
ParserInterface *NPInferenceSysBuilder::building_parser()
{

  LOG(INFO) << "Building parser....";
  return new OpenCVParser("OpenCV");
}