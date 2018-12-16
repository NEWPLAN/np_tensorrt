#ifndef __DATA_COLLECTOR_INTERFACE_H__
#define __DATA_COLLECTOR_INTERFACE_H__
#include <string>
#include <util/mylogs.h>
#include <util/blockingQueue.h>

class DataCollectorInterface
{
public:
  virtual std::string display() = 0;
  virtual void setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2]) = 0;
};
#endif
