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
    template<int matTypeV, class T, int B, int NB = mat::Dynamic>
    using Solver = SolverCholeskyDense<matTypeV, T, B, NB>;
  };

  template<>
  struct SolverTraits<solverCholeskySparse>
  {
    template<int matTypeV, class T, int B, int NB = mat::Dynamic>
    using Solver = SolverCholeskySparse<matTypeV, T, B, NB>;
  };

}