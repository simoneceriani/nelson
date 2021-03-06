#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFDDF : public Points2d3dBase < Points2d3dFDDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Dynamic, mat::Dynamic, numPoints3d > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFDDF_DenseDiagoDense = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_DenseDiagoDiago = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_DenseDiagoSpars = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_DenseDiagoSpaco = Points2d3dFDDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_DiagoDiagoDense = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_DiagoDiagoDiago = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_DiagoDiagoSpars = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_DiagoDiagoSpaco = Points2d3dFDDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SparsDiagoDense = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_SparsDiagoDiago = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_SparsDiagoSpars = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_SparsDiagoSpaco = Points2d3dFDDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDF_SpacoDiagoDense = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDF_SpacoDiagoDiago = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDF_SpacoDiagoSpars = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDF_SpacoDiagoSpaco = Points2d3dFDDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FDDF", "[DoubleSection-FDDF]",
  PointsSectionFDDF_DenseDiagoDense,
  PointsSectionFDDF_DenseDiagoDiago,
  PointsSectionFDDF_DenseDiagoSpars,
  PointsSectionFDDF_DenseDiagoSpaco,
  PointsSectionFDDF_DiagoDiagoDense,
  PointsSectionFDDF_DiagoDiagoDiago,
  PointsSectionFDDF_DiagoDiagoSpars,
  PointsSectionFDDF_DiagoDiagoSpaco,
  PointsSectionFDDF_SparsDiagoDense,
  PointsSectionFDDF_SparsDiagoDiago,
  PointsSectionFDDF_SparsDiagoSpars,
  PointsSectionFDDF_SparsDiagoSpaco,
  PointsSectionFDDF_SpacoDiagoDense,
  PointsSectionFDDF_SpacoDiagoDiago,
  PointsSectionFDDF_SpacoDiagoSpars,
  PointsSectionFDDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

