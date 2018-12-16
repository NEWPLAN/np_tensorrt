#ifndef __UTIL_BLOCKING_QUEUE_HPP__
#define __UTIL_BLOCKING_QUEUE_HPP__

#include <queue>
#include <string>
#include <vector>
#include <functional>
#include <condition_variable>
#include <util/mylogs.h>

template <typename T>
class BlockingQueue
{
public:
  BlockingQueue() {}
  virtual ~BlockingQueue() {}

  void push(const T &t)
  {
    {
      std::lock_guard<std::mutex> lg(mutex_);
      queue_.push(t);
    }
    condition_.notify_one();
  }
  // This logs a message if the threads needs to be blocked
  // useful for detecting e.g. when data feeding is too slow
  T pop(const char *log_on_wait)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty())
    {
      LOG_EVERY_N(INFO, 10000) << log_on_wait;
      condition_.wait(lock);
    }
    T t = queue_.front();
    queue_.pop();
    return t;
  }
  T pop()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty())
    {
      condition_.wait(lock);
    }
    T t(queue_.front());
    queue_.pop();
    return t;
  }

  bool try_peek(T *t)
  {
    std::lock_guard<std::mutex> lg(mutex_);
    if (queue_.empty())
    {
      return false;
    }
    *t = queue_.front();
    return true;
  }
  bool try_pop(T *t)
  {
    std::lock_guard<std::mutex> lg(mutex_);
    if (queue_.empty())
    {
      return false;
    }
    *t = queue_.front();
    queue_.pop();
    return true;
  }

  // Return element without removing it
  T peek()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty())
    {
      condition_.wait(lock);
    }
    return queue_.front();
  }

  size_t size() const
  {
    std::lock_guard<std::mutex> lg(mutex_);
    return queue_.size();
  }
  void pop_all(std::vector<T> &fetch_queue)
  {
    std::unique_lock<std::mutex> lg(mutex_);
    if (queue_.empty())
    {
      condition_.wait(lg);
    }
    while (!queue_.empty())
    {
      fetch_queue.push_back(std::move(queue_.front()));
      queue_.pop();
    }
  }

  bool nonblocking_push(const T &tt)
  {
    if (mutex_.try_lock())
    {
      queue_.push(tt);
      mutex_.unlock();
      return true;
    }
    return false;
  }

  bool
  nonblocking_size(size_t *size) const
  {
    if (mutex_.try_lock())
    {
      *size = queue_.size();
      mutex_.unlock();
      return true;
    }
    return false;
  }

protected:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

#endif
