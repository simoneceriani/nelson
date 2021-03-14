#include "SolverDiagonalBlocksInverseWWtMultSchur.h"
#include "SolverDiagonalBlocksInverseWWtMultSchur.hpp"

#include <sstream>
#include <iomanip>

namespace nelson {
  
    std::string SolverDiagonalBlocksInverseWWtMultSchurIterationTimeStat::toString(const std::string& linePrefix) const {
      std::ostringstream s;
      s << std::fixed << std::setprecision(6) << std::endl;
      s << linePrefix << "compute V^-1   = " << std::chrono::duration<double>(t1_VInvComputed - t0_startIteration).count() << std::endl;
      s << linePrefix << "compute W*V^-1 = " << std::chrono::duration<double>(t2_WVinvComputed - t1_VInvComputed).count() << std::endl;
      s << linePrefix << "compute bS     = " << std::chrono::duration<double>(t3_bSComputed - t2_WVinvComputed).count() << std::endl;
      s << linePrefix << "compute S      = " << std::chrono::duration<double>(t4_SComputed - t3_bSComputed).count() << std::endl;
      s << linePrefix << "xU = S \\ bs    = " << std::chrono::duration<double>(t5_xUSolved - t4_SComputed).count() << std::endl;
      s << linePrefix << "compute bVt    = " << std::chrono::duration<double>(t6_bVtildeComputed - t5_xUSolved).count() << std::endl;
      s << linePrefix << "compute bV     = " << std::chrono::duration<double>(t7_bVComputed - t6_bVtildeComputed).count() << std::endl;
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