#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVFD : public Points2d3dBase < Points2d3dFVFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, numPoints2d, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVFD() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

};

using PointsSectionFVFD_DenseDiagoDense = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_DenseDiagoDiago = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_DenseDiagoSpars = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_DenseDiagoSpaco = Points2d3dFVFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFD_DiagoDiagoDense = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_DiagoDiagoDiago = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_DiagoDiagoSpars = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_DiagoDiagoSpaco = Points2d3dFVFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFD_SparsDiagoDense = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_SparsDiagoDiago = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_SparsDiagoSpars = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_SparsDiagoSpaco = Points2d3dFVFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFD_SpacoDiagoDense = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFD_SpacoDiagoDiago = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFD_SpacoDiagoSpars = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFD_SpacoDiagoSpaco = Points2d3dFVFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FVFD", "[DoubleSection-FVFD]",
  PointsSectionFVFD_DenseDiagoDense,
  PointsSectionFVFD_DenseDiagoDiago,
  PointsSectionFVFD_DenseDiagoSpars,
  PointsSectionFVFD_DenseDiagoSpaco,
  PointsSectionFVFD_DiagoDiagoDense,
  PointsSectionFVFD_DiagoDiagoDiago,
  PointsSectionFVFD_DiagoDiagoSpars,
  PointsSectionFVFD_DiagoDiagoSpaco,
  PointsSectionFVFD_SparsDiagoDense,
  PointsSectionFVFD_SparsDiagoDiago,
  PointsSectionFVFD_SparsDiagoSpars,
  PointsSectionFVFD_SparsDiagoSpaco,
  PointsSectionFVFD_SpacoDiagoDense,
  PointsSectionFVFD_SpacoDiagoDiago,
  PointsSectionFVFD_SpacoDiagoSpars,
  PointsSectionFVFD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

