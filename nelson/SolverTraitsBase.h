#pragma once

#include "mat/Global.h"

#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"

namespace nelson {

  constexpr int solverCholeskyDense = 1;
  constexpr int solverCholeskySparse = 2;

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

}