/* Include files */

#include "Adaptive_EVBatteryCoolingSystem_cgxe.h"
#include "m_eLTcbc94yFfbvZfaWqHr1F.h"

unsigned int cgxe_Adaptive_EVBatteryCoolingSystem_method_dispatcher(SimStruct* S,
  int_T method, void* data)
{
  if (ssGetChecksum0(S) == 738221176 &&
      ssGetChecksum1(S) == 872787896 &&
      ssGetChecksum2(S) == 3929111085 &&
      ssGetChecksum3(S) == 2841443416) {
    method_dispatcher_eLTcbc94yFfbvZfaWqHr1F(S, method, data);
    return 1;
  }

  return 0;
}
