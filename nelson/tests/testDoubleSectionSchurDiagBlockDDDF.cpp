#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDDDF : public Points2d3dBase < Points2d3dDDDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Dynamic, mat::Dynamic, numPoints3d > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDDDF_DenseDiagoDense = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_DenseDiagoDiago = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_DenseDiagoSpars = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_DenseDiagoSpaco = Points2d3dDDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDF_DiagoDiagoDense = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_DiagoDiagoDiago = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_DiagoDiagoSpars = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_DiagoDiagoSpaco = Points2d3dDDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDF_SparsDiagoDense = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_SparsDiagoDiago = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_SparsDiagoSpars = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_SparsDiagoSpaco = Points2d3dDDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDF_SpacoDiagoDense = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDF_SpacoDiagoDiago = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDF_SpacoDiagoSpars = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDF_SpacoDiagoSpaco = Points2d3dDDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DDDF", "[DoubleSection-DDDF]",
  PointsSectionDDDF_DenseDiagoDense,
  PointsSectionDDDF_DenseDiagoDiago,
  PointsSectionDDDF_DenseDiagoSpars,
  PointsSectionDDDF_DenseDiagoSpaco,
  PointsSectionDDDF_DiagoDiagoDense,
  PointsSectionDDDF_DiagoDiagoDiago,
  PointsSectionDDDF_DiagoDiagoSpars,
  PointsSectionDDDF_DiagoDiagoSpaco,
  PointsSectionDDDF_SparsDiagoDense,
  PointsSectionDDDF_SparsDiagoDiago,
  PointsSectionDDDF_SparsDiagoSpars,
  PointsSectionDDDF_SparsDiagoSpaco,
  PointsSectionDDDF_SpacoDiagoDense,
  PointsSectionDDDF_SpacoDiagoDiago,
  PointsSectionDDDF_SpacoDiagoSpars,
  PointsSectionDDDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

