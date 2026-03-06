/* Include files */

#include "Adaptive_EVBatteryCoolingSystem_cgxe.h"
#include "m_p8GjkuxlZDFtXo89fdkiFF.h"

unsigned int cgxe_Adaptive_EVBatteryCoolingSystem_method_dispatcher(SimStruct* S,
  int_T method, void* data)
{
  if (ssGetChecksum0(S) == 4282087653 &&
      ssGetChecksum1(S) == 3913566072 &&
      ssGetChecksum2(S) == 2326703513 &&
      ssGetChecksum3(S) == 664292731) {
    method_dispatcher_p8GjkuxlZDFtXo89fdkiFF(S, method, data);
    return 1;
  }

  return 0;
}
