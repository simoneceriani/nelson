#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"

template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFFDD : public Points2d3dBase < Points2d3dFFDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, Point3d::blockSize, mat::Dynamic, mat::Dynamic > {

public:

  int numParametersV() const override {
    return numPoints3d;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFFDD_DenseDiagoDense = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DenseDiagoDiago = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DenseDiagoSpars = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DenseDiagoSpaco = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DiagoDiagoDense = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DiagoDiagoDiago = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DiagoDiagoSpars = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DiagoDiagoSpaco = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SparsDiagoDense = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SparsDiagoDiago = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SparsDiagoSpars = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SparsDiagoSpaco = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SpacoDiagoDense = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SpacoDiagoDiago = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SpacoDiagoSpars = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SpacoDiagoSpaco = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

using PointsSectionFFDD_DenseDiagoDense = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DenseDiagoDiago = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DenseDiagoSpars = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DenseDiagoSpaco = Points2d3dFFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_DiagoDiagoDense = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_DiagoDiagoDiago = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_DiagoDiagoSpars = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_DiagoDiagoSpaco = Points2d3dFFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SparsDiagoDense = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SparsDiagoDiago = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SparsDiagoSpars = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SparsDiagoSpaco = Points2d3dFFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFDD_SpacoDiagoDense = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFDD_SpacoDiagoDiago = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFDD_SpacoDiagoSpars = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFDD_SpacoDiagoSpaco = Points2d3dFFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FFDD", "[DoubleSection-FFDD]",
  PointsSectionFFDD_DenseDiagoDense,
  PointsSectionFFDD_DenseDiagoDiago,
  PointsSectionFFDD_DenseDiagoSpars,
  PointsSectionFFDD_DenseDiagoSpaco,
  PointsSectionFFDD_DiagoDiagoDense,
  PointsSectionFFDD_DiagoDiagoDiago,
  PointsSectionFFDD_DiagoDiagoSpars,
  PointsSectionFFDD_DiagoDiagoSpaco,
  PointsSectionFFDD_SparsDiagoDense,
  PointsSectionFFDD_SparsDiagoDiago,
  PointsSectionFFDD_SparsDiagoSpars,
  PointsSectionFFDD_SparsDiagoSpaco,
  PointsSectionFFDD_SpacoDiagoDense,
  PointsSectionFFDD_SpacoDiagoDiago,
  PointsSectionFFDD_SpacoDiagoSpars,
  PointsSectionFFDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}