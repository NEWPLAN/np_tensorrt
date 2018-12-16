#ifndef __INFERENCE_SYS_H__
#define __INFERENCE_SYS_H__

#include <data_collector_interface.h>
#include <engine_inferface.h>
#include <parser_inferface.h>

class InferenceSys
{
  private:
    DataCollectorInterface *data_collector_;
    EngineInterface *engine_;
    ParserInterface *parser_;

  public:
    virtual void set_engine(EngineInterface *eg)
    {
        this->engine_ = eg;
    }
    virtual EngineInterface *get_engine()
    {
        return this->engine_;
    }

    virtual void set_parser(ParserInterface *par)
    {
        this->parser_ = par;
    }
    virtual ParserInterface *get_parser()
    {
        return this->parser_;
    }

    virtual void set_dataCollector(DataCollectorInterface *dc)
    {
        this->data_collector_ = dc;
    }
    virtual DataCollectorInterface *get_dataCollector()
    {
        return this->data_collector_;
    }
};

#endif