#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"




template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVFDF : public Points2d3dBase < Points2d3dVFDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, Point3d::blockSize, mat::Dynamic, numPoints3d > {

  std::vector<int> u_sizes;
public:

  Points2d3dVFDF() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

};

using PointsSectionVFDF_DenseDiagoDense = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_DenseDiagoDiago = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_DenseDiagoSpars = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_DenseDiagoSpaco = Points2d3dVFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDF_DiagoDiagoDense = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_DiagoDiagoDiago = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_DiagoDiagoSpars = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_DiagoDiagoSpaco = Points2d3dVFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDF_SparsDiagoDense = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_SparsDiagoDiago = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_SparsDiagoSpars = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_SparsDiagoSpaco = Points2d3dVFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDF_SpacoDiagoDense = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDF_SpacoDiagoDiago = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDF_SpacoDiagoSpars = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDF_SpacoDiagoSpaco = Points2d3dVFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;



TEMPLATE_TEST_CASE("DoubleSection-VFDF", "[DoubleSection-VFDF]",
  PointsSectionVFDF_DenseDiagoDense,
  PointsSectionVFDF_DenseDiagoDiago,
  PointsSectionVFDF_DenseDiagoSpars,
  PointsSectionVFDF_DenseDiagoSpaco,
  PointsSectionVFDF_DiagoDiagoDense,
  PointsSectionVFDF_DiagoDiagoDiago,
  PointsSectionVFDF_DiagoDiagoSpars,
  PointsSectionVFDF_DiagoDiagoSpaco,
  PointsSectionVFDF_SparsDiagoDense,
  PointsSectionVFDF_SparsDiagoDiago,
  PointsSectionVFDF_SparsDiagoSpars,
  PointsSectionVFDF_SparsDiagoSpaco,
  PointsSectionVFDF_SpacoDiagoDense,
  PointsSectionVFDF_SpacoDiagoDiago,
  PointsSectionVFDF_SpacoDiagoSpars,
  PointsSectionVFDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

