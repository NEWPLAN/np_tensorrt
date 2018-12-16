#ifndef __PARSER_INTERFACE_H__
#define __PARSER_INTERFACE_H__
#include <string>
#include <util/blockingQueue.h>

class ParserInterface
{
public:
  virtual void setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2]) = 0;
  virtual std::string display() = 0;
};
#endif