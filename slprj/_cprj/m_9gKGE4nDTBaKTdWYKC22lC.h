#ifndef __9gKGE4nDTBaKTdWYKC22lC_h__
#define __9gKGE4nDTBaKTdWYKC22lC_h__

/* Include files */
#include "simstruc.h"
#include "rtwtypes.h"
#include "multiword_types.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_simstruct_bridge.h"
#include "sl_sfcn_cov/sl_sfcn_cov_bridge.h"

/* Type Definitions */
#ifndef typedef_InstanceStruct_9gKGE4nDTBaKTdWYKC22lC
#define typedef_InstanceStruct_9gKGE4nDTBaKTdWYKC22lC

typedef struct {
  SimStruct *S;
  PyObject *namespaceDict;
  PyGILState_STATE GIL;
  void *emlrtRootTLSGlobal;
  real_T *u0;
  real_T *u1;
  real_T *u2;
  real_T *b_y0;
  real_T *b_y1;
  real_T *y2;
  void **myController;
} InstanceStruct_9gKGE4nDTBaKTdWYKC22lC;

#endif                                 /* typedef_InstanceStruct_9gKGE4nDTBaKTdWYKC22lC */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */
extern void method_dispatcher_9gKGE4nDTBaKTdWYKC22lC(SimStruct *S, int_T method,
  void* data);

#endif
