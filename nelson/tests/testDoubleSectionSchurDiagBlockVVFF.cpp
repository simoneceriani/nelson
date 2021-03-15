#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVVFF : public Points2d3dBase < Points2d3dVVFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Variable, numPoints2d, numPoints3d > {
  std::vector<int> v_sizes;
  std::vector<int> u_sizes;
public:

  Points2d3dVVFF() : v_sizes(numPoints3d, Point3d::blockSize), u_sizes(numPoints2d, Point2d::blockSize) { }

  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


};

using PointsSectionVVFF_DenseDiagoDense = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_DenseDiagoDiago = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_DenseDiagoSpars = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_DenseDiagoSpaco = Points2d3dVVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFF_DiagoDiagoDense = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_DiagoDiagoDiago = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_DiagoDiagoSpars = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_DiagoDiagoSpaco = Points2d3dVVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFF_SparsDiagoDense = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_SparsDiagoDiago = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_SparsDiagoSpars = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_SparsDiagoSpaco = Points2d3dVVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVVFF_SpacoDiagoDense = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVVFF_SpacoDiagoDiago = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVVFF_SpacoDiagoSpars = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVVFF_SpacoDiagoSpaco = Points2d3dVVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-VVFF", "[DoubleSection-VVFF]",
  PointsSectionVVFF_DenseDiagoDense,
  PointsSectionVVFF_DenseDiagoDiago,
  PointsSectionVVFF_DenseDiagoSpars,
  PointsSectionVVFF_DenseDiagoSpaco,
  PointsSectionVVFF_DiagoDiagoDense,
  PointsSectionVVFF_DiagoDiagoDiago,
  PointsSectionVVFF_DiagoDiagoSpars,
  PointsSectionVVFF_DiagoDiagoSpaco,
  PointsSectionVVFF_SparsDiagoDense,
  PointsSectionVVFF_SparsDiagoDiago,
  PointsSectionVVFF_SparsDiagoSpars,
  PointsSectionVVFF_SparsDiagoSpaco,
  PointsSectionVVFF_SpacoDiagoDense,
  PointsSectionVVFF_SpacoDiagoDiago,
  PointsSectionVVFF_SpacoDiagoSpars,
  PointsSectionVVFF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

