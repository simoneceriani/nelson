#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFDDD : public Points2d3dBase < Points2d3dFDDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Dynamic, mat::Dynamic, mat::Dynamic > {

public:

  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFDDD_DenseDiagoDense = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_DenseDiagoDiago = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_DenseDiagoSpars = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_DenseDiagoSpaco = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DiagoDiagoDense = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_DiagoDiagoDiago = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_DiagoDiagoSpars = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_DiagoDiagoSpaco = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SparsDiagoDense = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_SparsDiagoDiago = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_SparsDiagoSpars = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_SparsDiagoSpaco = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SpacoDiagoDense = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_SpacoDiagoDiago = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_SpacoDiagoSpars = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_SpacoDiagoSpaco = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FDDD", "[DoubleSection-FDDD]",
  PointsSectionFDDD_DenseDiagoDense,
  PointsSectionFDDD_DenseDiagoDiago,
  PointsSectionFDDD_DenseDiagoSpars,
  PointsSectionFDDD_DenseDiagoSpaco,
  PointsSectionFDDD_DiagoDiagoDense,
  PointsSectionFDDD_DiagoDiagoDiago,
  PointsSectionFDDD_DiagoDiagoSpars,
  PointsSectionFDDD_DiagoDiagoSpaco,
  PointsSectionFDDD_SparsDiagoDense,
  PointsSectionFDDD_SparsDiagoDiago,
  PointsSectionFDDD_SparsDiagoSpars,
  PointsSectionFDDD_SparsDiagoSpaco,
  PointsSectionFDDD_SpacoDiagoDense,
  PointsSectionFDDD_SpacoDiagoDiago,
  PointsSectionFDDD_SpacoDiagoSpars,
  PointsSectionFDDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

