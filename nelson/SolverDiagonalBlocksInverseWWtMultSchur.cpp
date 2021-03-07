#include "SolverDiagonalBlocksInverseWWtMultSchur.h"
#include "SolverDiagonalBlocksInverseWWtMultSchur.hpp"

#include <sstream>
#include <iomanip>

namespace nelson {
  
    std::string SolverDiagonalBlocksInverseWWtMultSchurIterationTimeStat::toString(const std::string& linePrefix) const {
      std::ostringstream s;
      s << std::fixed << std::setprecision(6) << std::endl;
      s << linePrefix << "compute VInv = " << std::chrono::duration<double>(t1_VInvComputed - t0_startIteration).count() << std::endl;
      s << linePrefix << "VinvW        = " << std::chrono::duration<double>(t2_VinvWComputed - t1_VInvComputed).count() << std::endl;
      s << linePrefix << "WVinvWt      = " << std::chrono::duration<double>(t3_WVinvWtComputed - t2_VinvWComputed).count() << std::endl;
      s << linePrefix << "compute bS   = " << std::chrono::duration<double>(t4_bSComputed - t3_WVinvWtComputed).count() << std::endl;
      s << linePrefix << "compute S    = " << std::chrono::duration<double>(t5_SComputed - t4_bSComputed).count() << std::endl;
      s << linePrefix << "init chol S  = " << std::chrono::duration<double>(t6_SSolveInit - t5_SComputed).count() << std::endl;
      s << linePrefix << "factorize S  = " << std::chrono::duration<double>(t7_SFactorized - t6_SSolveInit).count() << std::endl;
      s << linePrefix << "solve bU     = " << std::chrono::duration<double>(t8_bUComputed - t7_SFactorized).count() << std::endl;
      s << linePrefix << "solve bVt    = " << std::chrono::duration<double>(t9_bVtildeComputed - t8_bUComputed).count() << std::endl;
      s << linePrefix << "solve bV     = " << std::chrono::duration<double>(t10_bVComputed - t9_bVtildeComputed).count() << std::endl;
      return s.str();
    }
  
    std::string SolverDiagonalBlocksInverseWWtMultSchurTimeStats::toString(const std::string& linePrefix) const {
      std::ostringstream s;
      s << std::fixed << std::setprecision(6) << std::endl;
      s << linePrefix << "init Time " << std::chrono::duration<double>(endInit - startInit).count() << std::endl;
      for (int i = 0; i < iterations.size(); i++) {
        s << linePrefix << "-- ITER " << i << " -- " << std::endl;
        s << linePrefix << iterations[i].toString("  ");
      }
      return s.str();
    }

}