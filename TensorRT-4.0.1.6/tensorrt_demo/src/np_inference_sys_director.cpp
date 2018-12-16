#include <np_inference_sys_director.h>

NPInferenceSysDirector::NPInferenceSysDirector(InferenceSysBuilder *build)
{
  this->builder_ = build;
}

InferenceSys *NPInferenceSysDirector::create_inferenceSys()
{
  EngineInterface *eg = builder_->building_engine();
  DataCollectorInterface *dc = builder_->building_dataCollector();
  ParserInterface *parser = builder_->building_parser();
  InferenceSys *infersys = new InferenceSys();

  infersys->set_engine(eg);
  infersys->set_dataCollector(dc);
  infersys->set_parser(parser);

  return infersys;
}