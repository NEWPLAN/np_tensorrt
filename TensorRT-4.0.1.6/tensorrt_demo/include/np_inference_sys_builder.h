#ifndef __NP_INFERENCE_SYS_BUILDER_H__
#define __NP_INFERENCE_SYS_BUILDER_H__

#include <engine_inferface.h>
#include <parser_inferface.h>
#include <data_collector_interface.h>
#include <inference_sys_builder.h>

class NPInferenceSysBuilder : public InferenceSysBuilder
{
public:
  virtual EngineInterface *building_engine();
  virtual DataCollectorInterface *building_dataCollector();
  virtual ParserInterface *building_parser();
};
#endif