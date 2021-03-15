#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dFVDF : public Points2d3dBase < Points2d3dFVDF<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, Point2d::blockSize, mat::Variable, mat::Dynamic, numPoints3d > {
  std::vector<int> v_sizes;
public:

  Points2d3dFVDF() : v_sizes(numPoints3d, Point3d::blockSize) { }

  const std::vector<int>& parameterVSize(void) const override {
    return v_sizes;
  }


  int numParametersU() const override {
    return numPoints2d;
  }

};

using PointsSectionFVDF_DenseDiagoDense = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_DenseDiagoDiago = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_DenseDiagoSpars = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_DenseDiagoSpaco = Points2d3dFVDF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDF_DiagoDiagoDense = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_DiagoDiagoDiago = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_DiagoDiagoSpars = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_DiagoDiagoSpaco = Points2d3dFVDF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDF_SparsDiagoDense = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_SparsDiagoDiago = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_SparsDiagoSpars = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_SparsDiagoSpaco = Points2d3dFVDF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFVDF_SpacoDiagoDense = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFVDF_SpacoDiagoDiago = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFVDF_SpacoDiagoSpars = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFVDF_SpacoDiagoSpaco = Points2d3dFVDF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FVDF", "[DoubleSection-FVDF]",
  PointsSectionFVDF_DenseDiagoDense,
  PointsSectionFVDF_DenseDiagoDiago,
  PointsSectionFVDF_DenseDiagoSpars,
  PointsSectionFVDF_DenseDiagoSpaco,
  PointsSectionFVDF_DiagoDiagoDense,
  PointsSectionFVDF_DiagoDiagoDiago,
  PointsSectionFVDF_DiagoDiagoSpars,
  PointsSectionFVDF_DiagoDiagoSpaco,
  PointsSectionFVDF_SparsDiagoDense,
  PointsSectionFVDF_SparsDiagoDiago,
  PointsSectionFVDF_SparsDiagoSpars,
  PointsSectionFVDF_SparsDiagoSpaco,
  PointsSectionFVDF_SpacoDiagoDense,
  PointsSectionFVDF_SpacoDiagoDiago,
  PointsSectionFVDF_SpacoDiagoSpars,
  PointsSectionFVDF_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

