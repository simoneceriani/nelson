#pragma once

#include "mat/Global.h"

#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"
#include "SolverCholeskyDenseSchur.h"

namespace nelson {

  constexpr int solverCholeskyDense = 1;
  constexpr int solverCholeskySparse = 2;
  constexpr int solverCholeskyDenseSchur = 3;

  template<int solverType>
  struct SolverTraits
  {

  };

  template<>
  struct SolverTraits<solverCholeskyDense>
  {
    template<class HessianTraits>
    using Solver = SolverCholeskyDense<HessianTraits::matType, typename HessianTraits::Type, HessianTraits::B, HessianTraits::NB>;
  };

  template<>
  struct SolverTraits<solverCholeskySparse>
  {
    template<class HessianTraits>
    using Solver = SolverCholeskySparse<HessianTraits::matType, typename HessianTraits::Type, HessianTraits::B, HessianTraits::NB>;
  };

  template<>
  struct SolverTraits<solverCholeskyDenseSchur>
  {
    template<class HessianTraits>
    using Solver = SolverCholeskyDenseSchur<
      HessianTraits::matTypeU, HessianTraits::matTypeV, HessianTraits::matTypeW,
      typename HessianTraits::Type, 
      HessianTraits::BU, HessianTraits::BV, 
      HessianTraits::NBU, HessianTraits::NBV
    >;
  };

}