#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVDDF : public Points2d3dBase < Points2d3dVDDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Dynamic, mat::Dynamic, numPoints3d > {

  std::vector<int> u_sizes;
public:

  Points2d3dVDDF() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }
  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

};

using PointsSectionVDDF_DenseDiagoDense = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_DenseDiagoDiago = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_DenseDiagoSpars = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_DenseDiagoSpaco = Points2d3dVDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDF_DiagoDiagoDense = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_DiagoDiagoDiago = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_DiagoDiagoSpars = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_DiagoDiagoSpaco = Points2d3dVDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDF_SparsDiagoDense = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_SparsDiagoDiago = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_SparsDiagoSpars = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_SparsDiagoSpaco = Points2d3dVDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDF_SpacoDiagoDense = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDF_SpacoDiagoDiago = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDF_SpacoDiagoSpars = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDF_SpacoDiagoSpaco = Points2d3dVDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-VDDF", "[DoubleSection-VDDF]",
  PointsSectionVDDF_DenseDiagoDense,
  PointsSectionVDDF_DenseDiagoDiago,
  PointsSectionVDDF_DenseDiagoSpars,
  PointsSectionVDDF_DenseDiagoSpaco,
  PointsSectionVDDF_DiagoDiagoDense,
  PointsSectionVDDF_DiagoDiagoDiago,
  PointsSectionVDDF_DiagoDiagoSpars,
  PointsSectionVDDF_DiagoDiagoSpaco,
  PointsSectionVDDF_SparsDiagoDense,
  PointsSectionVDDF_SparsDiagoDiago,
  PointsSectionVDDF_SparsDiagoSpars,
  PointsSectionVDDF_SparsDiagoSpaco,
  PointsSectionVDDF_SpacoDiagoDense,
  PointsSectionVDDF_SpacoDiagoDiago,
  PointsSectionVDDF_SpacoDiagoSpars,
  PointsSectionVDDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

