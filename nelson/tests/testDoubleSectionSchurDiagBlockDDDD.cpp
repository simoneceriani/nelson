#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDDDD : public Points2d3dBase < Points2d3dDDDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDDDD_DenseDiagoDense = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_DenseDiagoDiago = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_DenseDiagoSpars = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_DenseDiagoSpaco = Points2d3dDDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDD_DiagoDiagoDense = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_DiagoDiagoDiago = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_DiagoDiagoSpars = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_DiagoDiagoSpaco = Points2d3dDDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDD_SparsDiagoDense = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_SparsDiagoDiago = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_SparsDiagoSpars = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_SparsDiagoSpaco = Points2d3dDDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDDDD_SpacoDiagoDense = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDDDD_SpacoDiagoDiago = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDDDD_SpacoDiagoSpars = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDDDD_SpacoDiagoSpaco = Points2d3dDDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DDDD", "[DoubleSection-DDDD]",
  PointsSectionDDDD_DenseDiagoDense,
  PointsSectionDDDD_DenseDiagoDiago,
  PointsSectionDDDD_DenseDiagoSpars,
  PointsSectionDDDD_DenseDiagoSpaco,
  PointsSectionDDDD_DiagoDiagoDense,
  PointsSectionDDDD_DiagoDiagoDiago,
  PointsSectionDDDD_DiagoDiagoSpars,
  PointsSectionDDDD_DiagoDiagoSpaco,
  PointsSectionDDDD_SparsDiagoDense,
  PointsSectionDDDD_SparsDiagoDiago,
  PointsSectionDDDD_SparsDiagoSpars,
  PointsSectionDDDD_SparsDiagoSpaco,
  PointsSectionDDDD_SpacoDiagoDense,
  PointsSectionDDDD_SpacoDiagoDiago,
  PointsSectionDDDD_SpacoDiagoSpars,
  PointsSectionDDDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

