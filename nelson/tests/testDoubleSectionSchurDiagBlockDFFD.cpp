#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFFD : public Points2d3dBase < Points2d3dDFFD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, numPoints2d, mat::Dynamic > {

public:

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionDFFD_DenseDiagoDense = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_DenseDiagoDiago = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_DenseDiagoSpars = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_DenseDiagoSpaco = Points2d3dDFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFD_DiagoDiagoDense = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_DiagoDiagoDiago = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_DiagoDiagoSpars = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_DiagoDiagoSpaco = Points2d3dDFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFD_SparsDiagoDense = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_SparsDiagoDiago = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_SparsDiagoSpars = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_SparsDiagoSpaco = Points2d3dDFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFFD_SpacoDiagoDense = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFFD_SpacoDiagoDiago = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFFD_SpacoDiagoSpars = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFFD_SpacoDiagoSpaco = Points2d3dDFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-DFFD", "[DoubleSection-DFFD]",
  PointsSectionDFFD_DenseDiagoDense,
  PointsSectionDFFD_DenseDiagoDiago,
  PointsSectionDFFD_DenseDiagoSpars,
  PointsSectionDFFD_DenseDiagoSpaco,
  PointsSectionDFFD_DiagoDiagoDense,
  PointsSectionDFFD_DiagoDiagoDiago,
  PointsSectionDFFD_DiagoDiagoSpars,
  PointsSectionDFFD_DiagoDiagoSpaco,
  PointsSectionDFFD_SparsDiagoDense,
  PointsSectionDFFD_SparsDiagoDiago,
  PointsSectionDFFD_SparsDiagoSpars,
  PointsSectionDFFD_SparsDiagoSpaco,
  PointsSectionDFFD_SpacoDiagoDense,
  PointsSectionDFFD_SpacoDiagoDiago,
  PointsSectionDFFD_SpacoDiagoSpars,
  PointsSectionDFFD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

