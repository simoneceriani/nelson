#pragma once

#include "mat/Global.h"

#include "SolverTraitsBase.h"
#include "SolverCholeskySchur.h"
#include "SolverDiagonalBlocksInverseSchur.h"
#include "SolverDiagonalBlocksInverseWWtMultSchur.h"

namespace nelson {

  constexpr int solverCholeskySchur = 3;
  constexpr int solverCholeskySchurDiagBlockInverse = 4;
  constexpr int solverCholeskySchurDiagBlockInverseWWtMult = 5;

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

  template<>
  struct SolverTraits<solverCholeskySchurDiagBlockInverseWWtMult>
  {
    template<class HessianTraits, int SType, int VinvType, int choleskyOrderingS>
    using Solver = SolverDiagonalBlocksInverseWWtMultSchur<
      HessianTraits::matTypeU, HessianTraits::matTypeW,
      typename HessianTraits::Type,
      HessianTraits::BU, HessianTraits::BV,
      HessianTraits::NBU, HessianTraits::NBV,
      SType, VinvType,
      choleskyOrderingS
    >;
  };

}