#pragma once

#include "mat/Global.h"

#include "SolverCholeskyDense.h"
#include "SolverCholeskySparse.h"
#include "SolverCholeskySchur.h"

namespace nelson {

  constexpr int solverCholeskyDense = 1;
  constexpr int solverCholeskySparse = 2;
  constexpr int solverCholeskySchurDense = 3;
  constexpr int solverCholeskySchurSparse = 4;

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
  struct SolverTraits<solverCholeskySchurDense>
  {
    template<class HessianTraits>
    using Solver = SolverCholeskySchur<
      HessianTraits::matTypeU, HessianTraits::matTypeV, HessianTraits::matTypeW,
      typename HessianTraits::Type,
      HessianTraits::BU, HessianTraits::BV,
      HessianTraits::NBU, HessianTraits::NBV,
      SolverCholeskyDense<HessianTraits::matTypeU, typename HessianTraits::Type, HessianTraits::BU, HessianTraits::NBU>
    >;
  };

  template<>
  struct SolverTraits<solverCholeskySchurSparse>
  {
    template<class HessianTraits>
    using Solver = SolverCholeskySchur<
      HessianTraits::matTypeU, HessianTraits::matTypeV, HessianTraits::matTypeW,
      typename HessianTraits::Type,
      HessianTraits::BU, HessianTraits::BV,
      HessianTraits::NBU, HessianTraits::NBV,
      SolverCholeskySparse<HessianTraits::matTypeU, typename HessianTraits::Type, HessianTraits::BU, HessianTraits::NBU>
    >;
  };

}