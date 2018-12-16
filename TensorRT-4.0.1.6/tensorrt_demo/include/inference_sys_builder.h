#ifndef __INFERENCE_BUILDER_H__
#define __INFERENCE_BUILDER_H__
#include <inference_sys.h>
#include <engine_inferface.h>
#include <parser_inferface.h>
#include <data_collector_interface.h>

class InferenceSysBuilder
{
  public:
    virtual EngineInterface *building_engine() = 0;
    virtual DataCollectorInterface *building_dataCollector() = 0;
    virtual ParserInterface *building_parser() = 0;
};
#endif