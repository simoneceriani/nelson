#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVFFF : public Points2d3dBase < Points2d3dVFFF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, Point3d::blockSize, numPoints2d, numPoints3d > {
  std::vector<int> u_sizes;
public:

  Points2d3dVFFF() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }


};

using PointsSectionVFFF_DenseDiagoDense = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_DenseDiagoDiago = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_DenseDiagoSpars = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_DenseDiagoSpaco = Points2d3dVFFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFF_DiagoDiagoDense = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_DiagoDiagoDiago = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_DiagoDiagoSpars = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_DiagoDiagoSpaco = Points2d3dVFFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFF_SparsDiagoDense = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_SparsDiagoDiago = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_SparsDiagoSpars = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_SparsDiagoSpaco = Points2d3dVFFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFFF_SpacoDiagoDense = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFFF_SpacoDiagoDiago = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFFF_SpacoDiagoSpars = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFFF_SpacoDiagoSpaco = Points2d3dVFFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-VFFF", "[DoubleSection-VFFF]",
  PointsSectionVFFF_DenseDiagoDense,
  PointsSectionVFFF_DenseDiagoDiago,
  PointsSectionVFFF_DenseDiagoSpars,
  PointsSectionVFFF_DenseDiagoSpaco,
  PointsSectionVFFF_DiagoDiagoDense,
  PointsSectionVFFF_DiagoDiagoDiago,
  PointsSectionVFFF_DiagoDiagoSpars,
  PointsSectionVFFF_DiagoDiagoSpaco,
  PointsSectionVFFF_SparsDiagoDense,
  PointsSectionVFFF_SparsDiagoDiago,
  PointsSectionVFFF_SparsDiagoSpars,
  PointsSectionVFFF_SparsDiagoSpaco,
  PointsSectionVFFF_SpacoDiagoDense,
  PointsSectionVFFF_SpacoDiagoDiago,
  PointsSectionVFFF_SpacoDiagoSpars,
  PointsSectionVFFF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

