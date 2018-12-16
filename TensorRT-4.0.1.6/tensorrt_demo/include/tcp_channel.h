#ifndef __DATA_COLLECTOR_FROM_TCP_H__
#define __DATA_COLLECTOR_FROM_TCP_H__
#include <string>
#include <util/mylogs.h>
#include <data_collector_interface.h>
#include <vector>
#include <thread>

class TCPChannel : public DataCollectorInterface
{
public:
  TCPChannel(std::string nm);
  ~TCPChannel();

  virtual std::string display();
  virtual void setup(BlockingQueue<int> *from[2], BlockingQueue<int> *to[2]);

  void client_handler(int new_fd, BlockingQueue<int> *to[2]);

private:
  std::string name_;
  int listen_fd;
  std::string ip_addr;
  int port;
  std::vector<std::thread *> _handler;
};
#endif
