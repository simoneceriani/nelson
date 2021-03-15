#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dDVDD : public Points2d3dBase < Points2d3dDVDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dDVDD() : v_sizes(numPoints3d, Point3d::blockSize) { }

  int parameterUSize(void) const override {
    return Point2d::blockSize;
  }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionDVDD_DenseDiagoDense = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_DenseDiagoDiago = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_DenseDiagoSpars = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_DenseDiagoSpaco = Points2d3dDVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDD_DiagoDiagoDense = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_DiagoDiagoDiago = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_DiagoDiagoSpars = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_DiagoDiagoSpaco = Points2d3dDVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDD_SparsDiagoDense = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_SparsDiagoDiago = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_SparsDiagoSpars = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_SparsDiagoSpaco = Points2d3dDVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionDVDD_SpacoDiagoDense = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionDVDD_SpacoDiagoDiago = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionDVDD_SpacoDiagoSpars = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionDVDD_SpacoDiagoSpaco = Points2d3dDVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-DVDD", "[DoubleSection-DVDD]",
  PointsSectionDVDD_DenseDiagoDense,
  PointsSectionDVDD_DenseDiagoDiago,
  PointsSectionDVDD_DenseDiagoSpars,
  PointsSectionDVDD_DenseDiagoSpaco,
  PointsSectionDVDD_DiagoDiagoDense,
  PointsSectionDVDD_DiagoDiagoDiago,
  PointsSectionDVDD_DiagoDiagoSpars,
  PointsSectionDVDD_DiagoDiagoSpaco,
  PointsSectionDVDD_SparsDiagoDense,
  PointsSectionDVDD_SparsDiagoDiago,
  PointsSectionDVDD_SparsDiagoSpars,
  PointsSectionDVDD_SparsDiagoSpaco,
  PointsSectionDVDD_SpacoDiagoDense,
  PointsSectionDVDD_SpacoDiagoDiago,
  PointsSectionDVDD_SpacoDiagoSpars,
  PointsSectionDVDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

