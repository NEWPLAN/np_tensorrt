#ifndef __ENGINE_INTERFACE_H__
#define __ENGINE_INTERFACE_H__
#include <string>
#include <util/mylogs.h>
#include <util/blockingQueue.h>

class EngineInterface
{
public:
  virtual void setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2]) = 0;
  virtual std::string display() = 0;
};

#endif