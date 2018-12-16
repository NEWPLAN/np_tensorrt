#ifndef __INFERENCE_SYS_DIRRECTOR_H__
#define __INFERENCE_SYS_DIRRECTOR_H__
#include <inference_sys.h>

class InferenceSysDirector
{
  public:
    virtual InferenceSys *create_inferenceSys() = 0;
};

#endif