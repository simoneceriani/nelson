#pragma once

#include "mat/Global.h"
#include <Eigen/OrderingMethods>


namespace nelson {

  constexpr int choleskyNaturalOrdering = 0;
  constexpr int choleskyAMDOrdering = 1;
  constexpr int choleskyCOLAMDOrdering = 2;

  template <int Ordering>
  struct CholeskyOrderingTraits {

  };

  template<>
  struct CholeskyOrderingTraits<choleskyNaturalOrdering>
  {
    using Ordering = Eigen::NaturalOrdering<int>;
  };

  template<>
  struct CholeskyOrderingTraits<choleskyAMDOrdering>
  {
    using Ordering = Eigen::AMDOrdering<int>;
  };

  template<>
  struct CholeskyOrderingTraits<choleskyCOLAMDOrdering>
  {
    using Ordering = Eigen::COLAMDOrdering<int>;
  };


}