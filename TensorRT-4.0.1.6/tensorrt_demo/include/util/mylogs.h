#ifndef __MYLOG_H__
#define __MYLOG_H__
#include <glog/logging.h>
class log_sys
{
  public:
    explicit log_sys(const char *exe_name)
    {
        if (exe_name == nullptr)
        {
            exit(0);
        }
        google::InitGoogleLogging(exe_name);
        FLAGS_stderrthreshold = google::INFO;
        FLAGS_colorlogtostderr = true;
        LOG(INFO) << "test ............ log";
    }
    virtual ~log_sys()
    {
        LOG(INFO) << "will return and exit";
        google::ShutdownGoogleLogging();
    }
};

#endif