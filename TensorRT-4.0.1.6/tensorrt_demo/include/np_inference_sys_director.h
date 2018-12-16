#ifndef __NP_INFERENCE_SYS_DIRRECTOR_H__
#define __NP_INFERENCE_SYS_DIRRECTOR_H__
#include <inference_sys.h>
#include <inference_sys_director.h>
#include <inference_sys_builder.h>

class NPInferenceSysDirector : public InferenceSysDirector
{
public:
  NPInferenceSysDirector(InferenceSysBuilder *build);
  virtual InferenceSys *create_inferenceSys();

private:
  InferenceSysBuilder *builder_;
};

#endif