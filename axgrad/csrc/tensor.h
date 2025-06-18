/**
  @file tensor.h header file for tensor.cpp & tensor
  * contains all the tensor related ops
  * imports basic core & basic functionalities from core/core.h
  * cpu based helper codes from cpu/
  * cuda based codes from cuda/
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtensor.so core/core.cpp core/dtype.cpp tensor.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp cpu/red_ops.cpp cpu/binary_ops.cpp
    *- '.dll': g++ -shared -o libtensor.dll core/core.cpp core/dtype.cpp tensor.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp cpu/red_ops.cpp cpu/binary_ops.cpp
    *- '.dylib': g++ -dynamiclib -o libtensor.dylib core/core.cpp core/dtype.cpp tensor.cpp cpu/maths_ops.cpp cpu/helpers.cpp cpu/utils.cpp cpu/red_ops.cpp cpu/binary_ops.cpp
*/

#ifndef __TENSOR__H__
#define __TENSOR__H__

#include <stdlib.h>
#include "core/core.h"

extern "C" {
  // maths ops
}

#endif  //!__TENSOR__H__