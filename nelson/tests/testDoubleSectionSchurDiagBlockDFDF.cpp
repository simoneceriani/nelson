#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFDF : public Points2d3dBase < Points2d3dDFDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, mat::Dynamic, numPoints3d > {

public:

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDFDF_DenseDiagoDense = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_DenseDiagoDiago = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_DenseDiagoSpars = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_DenseDiagoSpaco = Points2d3dDFDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDF_DiagoDiagoDense = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_DiagoDiagoDiago = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_DiagoDiagoSpars = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_DiagoDiagoSpaco = Points2d3dDFDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDF_SparsDiagoDense = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_SparsDiagoDiago = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_SparsDiagoSpars = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_SparsDiagoSpaco = Points2d3dDFDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDF_SpacoDiagoDense = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDF_SpacoDiagoDiago = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDF_SpacoDiagoSpars = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDF_SpacoDiagoSpaco = Points2d3dDFDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-DFDF", "[DoubleSection-DFDF]",
  PointsSectionDFDF_DenseDiagoDense,
  PointsSectionDFDF_DenseDiagoDiago,
  PointsSectionDFDF_DenseDiagoSpars,
  PointsSectionDFDF_DenseDiagoSpaco,
  PointsSectionDFDF_DiagoDiagoDense,
  PointsSectionDFDF_DiagoDiagoDiago,
  PointsSectionDFDF_DiagoDiagoSpars,
  PointsSectionDFDF_DiagoDiagoSpaco,
  PointsSectionDFDF_SparsDiagoDense,
  PointsSectionDFDF_SparsDiagoDiago,
  PointsSectionDFDF_SparsDiagoSpars,
  PointsSectionDFDF_SparsDiagoSpaco,
  PointsSectionDFDF_SpacoDiagoDense,
  PointsSectionDFDF_SpacoDiagoDiago,
  PointsSectionDFDF_SpacoDiagoSpars,
  PointsSectionDFDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

