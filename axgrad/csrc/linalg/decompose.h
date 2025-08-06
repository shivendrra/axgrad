#ifndef __DECOMPOSE__H__
#define __DECOMPOSE__H__

#include "../core/core.h"
#include "../core/dtype.h"

extern "C" {
  Tensor** svd_tensor(Tensor* a);
  Tensor* cholesky_tensor(Tensor* a);
  Tensor* eig_tensor(Tensor* a);        // eigen values
  Tensor* eigv_tensor(Tensor* a);        // eigen vectors
  Tensor* eigh_tensor(Tensor* a);        // eigen hermitian values
  Tensor* eighv_tensor(Tensor* a);         // eigen hermitian vectors
  Tensor* batched_eig_tensor(Tensor* a);    // batched eigen values
  Tensor* batched_eigv_tensor(Tensor* a);    // batched eigen vectors
  Tensor* batched_eigh_tensor(Tensor* a);      // batched eigen hermitian values
  Tensor* batched_eighv_tensor(Tensor* a);      // batched eigen hermitian vectors
  Tensor** qr_tensor(Tensor* a);
  Tensor** batched_qr_tensor(Tensor* a);
  Tensor** lu_tensor(Tensor* a);
  Tensor** batched_lu_tensor(Tensor* a);
}

#endif  //!__DECOMPOSE__H__