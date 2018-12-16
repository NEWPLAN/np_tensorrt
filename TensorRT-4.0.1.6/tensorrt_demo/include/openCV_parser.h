#ifndef __OPENCV_PARSER_H__
#define __OPENCV_PARSER_H__
#include <parser_inferface.h>
#include <util/mylogs.h>
#include <util/blockingQueue.h>

class DecoderHandler;

class OpenCVParser : public ParserInterface
{
public:
  OpenCVParser(std::string nm);
  virtual ~OpenCVParser();

  virtual void setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2]);
  virtual std::string display();

private:
  std::string name_;
  std::vector<DecoderHandler *> hder;
};
#endif