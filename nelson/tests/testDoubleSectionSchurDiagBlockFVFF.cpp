#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVFF : public Points2d3dBase < Points2d3dFVFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, numPoints2d, numPoints3d > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVFF() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


};

using PointsSectionFVFF_DenseDiagoDense = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_DenseDiagoDiago = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_DenseDiagoSpars = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_DenseDiagoSpaco = Points2d3dFVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFF_DiagoDiagoDense = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_DiagoDiagoDiago = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_DiagoDiagoSpars = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_DiagoDiagoSpaco = Points2d3dFVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFF_SparsDiagoDense = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_SparsDiagoDiago = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_SparsDiagoSpars = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_SparsDiagoSpaco = Points2d3dFVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVFF_SpacoDiagoDense = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVFF_SpacoDiagoDiago = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVFF_SpacoDiagoSpars = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVFF_SpacoDiagoSpaco = Points2d3dFVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FVFF", "[DoubleSection-FVFF]",
  PointsSectionFVFF_DenseDiagoDense,
  PointsSectionFVFF_DenseDiagoDiago,
  PointsSectionFVFF_DenseDiagoSpars,
  PointsSectionFVFF_DenseDiagoSpaco,
  PointsSectionFVFF_DiagoDiagoDense,
  PointsSectionFVFF_DiagoDiagoDiago,
  PointsSectionFVFF_DiagoDiagoSpars,
  PointsSectionFVFF_DiagoDiagoSpaco,
  PointsSectionFVFF_SparsDiagoDense,
  PointsSectionFVFF_SparsDiagoDiago,
  PointsSectionFVFF_SparsDiagoSpars,
  PointsSectionFVFF_SparsDiagoSpaco,
  PointsSectionFVFF_SpacoDiagoDense,
  PointsSectionFVFF_SpacoDiagoDiago,
  PointsSectionFVFF_SpacoDiagoSpars,
  PointsSectionFVFF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

