#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVDD : public Points2d3dBase < Points2d3dFVDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, mat::Dynamic, mat::Dynamic > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVDD() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }

  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFVDD_DenseDiagoDense = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_DenseDiagoDiago = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_DenseDiagoSpars = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_DenseDiagoSpaco = Points2d3dFVDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDD_DiagoDiagoDense = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_DiagoDiagoDiago = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_DiagoDiagoSpars = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_DiagoDiagoSpaco = Points2d3dFVDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDD_SparsDiagoDense = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_SparsDiagoDiago = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_SparsDiagoSpars = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_SparsDiagoSpaco = Points2d3dFVDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDD_SpacoDiagoDense = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDD_SpacoDiagoDiago = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDD_SpacoDiagoSpars = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDD_SpacoDiagoSpaco = Points2d3dFVDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-FVDD", "[DoubleSection-FVDD]",
  PointsSectionFVDD_DenseDiagoDense,
  PointsSectionFVDD_DenseDiagoDiago,
  PointsSectionFVDD_DenseDiagoSpars,
  PointsSectionFVDD_DenseDiagoSpaco,
  PointsSectionFVDD_DiagoDiagoDense,
  PointsSectionFVDD_DiagoDiagoDiago,
  PointsSectionFVDD_DiagoDiagoSpars,
  PointsSectionFVDD_DiagoDiagoSpaco,
  PointsSectionFVDD_SparsDiagoDense,
  PointsSectionFVDD_SparsDiagoDiago,
  PointsSectionFVDD_SparsDiagoSpars,
  PointsSectionFVDD_SparsDiagoSpaco,
  PointsSectionFVDD_SpacoDiagoDense,
  PointsSectionFVDD_SpacoDiagoDiago,
  PointsSectionFVDD_SpacoDiagoSpars,
  PointsSectionFVDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

