#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDFDD : public Points2d3dBase < Points2d3dDFDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, Point3d::blockSize, mat::Dynamic, mat::Dynamic > {

public:

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

using PointsSectionDFDD_DenseDiagoDense = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_DenseDiagoDiago = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_DenseDiagoSpars = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_DenseDiagoSpaco = Points2d3dDFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDD_DiagoDiagoDense = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_DiagoDiagoDiago = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_DiagoDiagoSpars = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_DiagoDiagoSpaco = Points2d3dDFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDD_SparsDiagoDense = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_SparsDiagoDiago = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_SparsDiagoSpars = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_SparsDiagoSpaco = Points2d3dDFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDFDD_SpacoDiagoDense = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDFDD_SpacoDiagoDiago = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDFDD_SpacoDiagoSpars = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDFDD_SpacoDiagoSpaco = Points2d3dDFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DFDD", "[DoubleSection-DFDD]",
  PointsSectionDFDD_DenseDiagoDense,
  PointsSectionDFDD_DenseDiagoDiago,
  PointsSectionDFDD_DenseDiagoSpars,
  PointsSectionDFDD_DenseDiagoSpaco,
  PointsSectionDFDD_DiagoDiagoDense,
  PointsSectionDFDD_DiagoDiagoDiago,
  PointsSectionDFDD_DiagoDiagoSpars,
  PointsSectionDFDD_DiagoDiagoSpaco,
  PointsSectionDFDD_SparsDiagoDense,
  PointsSectionDFDD_SparsDiagoDiago,
  PointsSectionDFDD_SparsDiagoSpars,
  PointsSectionDFDD_SparsDiagoSpaco,
  PointsSectionDFDD_SpacoDiagoDense,
  PointsSectionDFDD_SpacoDiagoDiago,
  PointsSectionDFDD_SpacoDiagoSpars,
  PointsSectionDFDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

