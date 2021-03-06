#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDVFF : public Points2d3dBase < Points2d3dDVFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Variable, numPoints2d, numPoints3d > {
  std::vector<int> v_sizes;
public:

  Points2d3dDVFF() : v_sizes(numPoints3d, Point3d::blockSize) { }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


};

using PointsSectionDVFF_DenseDiagoDense = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_DenseDiagoDiago = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_DenseDiagoSpars = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_DenseDiagoSpaco = Points2d3dDVFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFF_DiagoDiagoDense = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_DiagoDiagoDiago = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_DiagoDiagoSpars = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_DiagoDiagoSpaco = Points2d3dDVFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFF_SparsDiagoDense = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_SparsDiagoDiago = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_SparsDiagoSpars = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_SparsDiagoSpaco = Points2d3dDVFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVFF_SpacoDiagoDense = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVFF_SpacoDiagoDiago = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVFF_SpacoDiagoSpars = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVFF_SpacoDiagoSpaco = Points2d3dDVFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DVFF", "[DoubleSection-DVFF]",
  PointsSectionDVFF_DenseDiagoDense,
  PointsSectionDVFF_DenseDiagoDiago,
  PointsSectionDVFF_DenseDiagoSpars,
  PointsSectionDVFF_DenseDiagoSpaco,
  PointsSectionDVFF_DiagoDiagoDense,
  PointsSectionDVFF_DiagoDiagoDiago,
  PointsSectionDVFF_DiagoDiagoSpars,
  PointsSectionDVFF_DiagoDiagoSpaco,
  PointsSectionDVFF_SparsDiagoDense,
  PointsSectionDVFF_SparsDiagoDiago,
  PointsSectionDVFF_SparsDiagoSpars,
  PointsSectionDVFF_SparsDiagoSpaco,
  PointsSectionDVFF_SpacoDiagoDense,
  PointsSectionDVFF_SpacoDiagoDiago,
  PointsSectionDVFF_SpacoDiagoSpars,
  PointsSectionDVFF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

