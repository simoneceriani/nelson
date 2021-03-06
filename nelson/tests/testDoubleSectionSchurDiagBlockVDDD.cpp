#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"



template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVDDD : public Points2d3dBase < Points2d3dVDDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic > {

  std::vector<int> u_sizes;
public:

  Points2d3dVDDD() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }
  int parameterVSize(void) const override {
    return Point3d::blockSize;
  }


  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionVDDD_DenseDiagoDense = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_DenseDiagoDiago = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_DenseDiagoSpars = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_DenseDiagoSpaco = Points2d3dVDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDD_DiagoDiagoDense = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_DiagoDiagoDiago = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_DiagoDiagoSpars = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_DiagoDiagoSpaco = Points2d3dVDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDD_SparsDiagoDense = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_SparsDiagoDiago = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_SparsDiagoSpars = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_SparsDiagoSpaco = Points2d3dVDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVDDD_SpacoDiagoDense = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVDDD_SpacoDiagoDiago = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVDDD_SpacoDiagoSpars = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVDDD_SpacoDiagoSpaco = Points2d3dVDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-VDDD", "[DoubleSection-VDDD]",
  PointsSectionVDDD_DenseDiagoDense,
  PointsSectionVDDD_DenseDiagoDiago,
  PointsSectionVDDD_DenseDiagoSpars,
  PointsSectionVDDD_DenseDiagoSpaco,
  PointsSectionVDDD_DiagoDiagoDense,
  PointsSectionVDDD_DiagoDiagoDiago,
  PointsSectionVDDD_DiagoDiagoSpars,
  PointsSectionVDDD_DiagoDiagoSpaco,
  PointsSectionVDDD_SparsDiagoDense,
  PointsSectionVDDD_SparsDiagoDiago,
  PointsSectionVDDD_SparsDiagoSpars,
  PointsSectionVDDD_SparsDiagoSpaco,
  PointsSectionVDDD_SpacoDiagoDense,
  PointsSectionVDDD_SpacoDiagoDiago,
  PointsSectionVDDD_SpacoDiagoSpars,
  PointsSectionVDDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

