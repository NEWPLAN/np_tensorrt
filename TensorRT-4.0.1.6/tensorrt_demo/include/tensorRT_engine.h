#ifndef __TENSORRT_ENGINE_H__
#define __TENSORRT_ENGINE_H__

#include <string>
#include <util/mylogs.h>
#include <engine_inferface.h>
#include <vector>

class TensorRTEngine : public EngineInterface
{
public:
  TensorRTEngine(std::string nm);
  virtual ~TensorRTEngine();

  virtual void setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2]);
  virtual std::string display();

private:
  std::string name_;
};

#endif
