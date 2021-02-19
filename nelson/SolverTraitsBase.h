#pragma once

#include "mat/Global.h"

#include "OrderingTraits.h"

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
    template<class HessianTraits, int choleskyOrderingIgnored = -1>
    using Solver = SolverCholeskyDense<HessianTraits::matType, typename HessianTraits::Type, HessianTraits::B, HessianTraits::NB>;

    template<class EigenMat, int choleskyOrderingIgnored = -1>
    using SolverEigen = SolverCholeskyEigenDense<EigenMat>;

  };

  template<>
  struct SolverTraits<solverCholeskySparse>
  {
    template<class HessianTraits, int choleskyOrdering>
    using Solver = SolverCholeskySparse<HessianTraits::matType, typename HessianTraits::Type, HessianTraits::B, HessianTraits::NB, typename CholeskyOrderingTraits<choleskyOrdering>::Ordering>;

    template<class EigenMat, int choleskyOrdering>
    using SolverEigen = SolverCholeskyEigenSparse<EigenMat, typename CholeskyOrderingTraits<choleskyOrdering>::Ordering>;
  };

}