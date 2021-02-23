#pragma once

#include "mat/Global.h"

#include "SolverTraitsBase.h"
#include "SolverCholeskySchur.h"
#include "SolverDiagonalBlocksInverseSchur.h"

namespace nelson {

  constexpr int solverCholeskySchur = 3;
  constexpr int solverCholeskySchurDiagBlockInverse = 4;

  template<>
  struct SolverTraits<solverCholeskySchur>
  {
    template<class HessianTraits, int wrapperUType, int wrapperWType, int solverVType, int choleskyOrderingS, int choleskyOrderingV>
    using Solver = SolverCholeskySchur<
      HessianTraits::matTypeU, HessianTraits::matTypeV, HessianTraits::matTypeW,
      typename HessianTraits::Type,
      HessianTraits::BU, HessianTraits::BV,
      HessianTraits::NBU, HessianTraits::NBV,
      wrapperUType, wrapperWType, solverVType,
      choleskyOrderingS, choleskyOrderingV
    >;
  };

  template<>
  struct SolverTraits<solverCholeskySchurDiagBlockInverse>
  {
    template<class HessianTraits, int wrapperUType, int wrapperWType, int choleskyOrderingS>
    using Solver = SolverDiagonalBlocksInverseSchur<
      HessianTraits::matTypeU, HessianTraits::matTypeW,
      typename HessianTraits::Type,
      HessianTraits::BU, HessianTraits::BV,
      HessianTraits::NBU, HessianTraits::NBV,
      wrapperUType, wrapperWType,
      choleskyOrderingS
    >;
  };

}