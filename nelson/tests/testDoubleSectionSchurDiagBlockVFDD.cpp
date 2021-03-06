#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "testDoubleSectionSchurCommon.hpp"


template<int matTypeUv, int matTypeVv, int matTypeWv>
class Points2d3dVFDD : public Points2d3dBase < Points2d3dVFDD<matTypeUv, matTypeVv, matTypeWv>, matTypeUv, matTypeVv, matTypeWv, mat::Variable, Point3d::blockSize, mat::Dynamic, mat::Dynamic > {

  std::vector<int> u_sizes;
public:

  Points2d3dVFDD() : u_sizes(numPoints2d, Point2d::blockSize) { }
  const std::vector<int>& parameterUSize(void) const override {
    return u_sizes;
  }

  int numParametersV() const override {
    return numPoints3d;
  }

};

using PointsSectionVFDD_DenseDiagoDense = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_DenseDiagoDiago = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_DenseDiagoSpars = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_DenseDiagoSpaco = Points2d3dVFDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDD_DiagoDiagoDense = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_DiagoDiagoDiago = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_DiagoDiagoSpars = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_DiagoDiagoSpaco = Points2d3dVFDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDD_SparsDiagoDense = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_SparsDiagoDiago = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_SparsDiagoSpars = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_SparsDiagoSpaco = Points2d3dVFDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionVFDD_SpacoDiagoDense = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionVFDD_SpacoDiagoDiago = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionVFDD_SpacoDiagoSpars = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionVFDD_SpacoDiagoSpaco = Points2d3dVFDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("DoubleSection-VFDD", "[DoubleSection-VFDD]",
  PointsSectionVFDD_DenseDiagoDense,
  PointsSectionVFDD_DenseDiagoDiago,
  PointsSectionVFDD_DenseDiagoSpars,
  PointsSectionVFDD_DenseDiagoSpaco,
  PointsSectionVFDD_DiagoDiagoDense,
  PointsSectionVFDD_DiagoDiagoDiago,
  PointsSectionVFDD_DiagoDiagoSpars,
  PointsSectionVFDD_DiagoDiagoSpaco,
  PointsSectionVFDD_SparsDiagoDense,
  PointsSectionVFDD_SparsDiagoDiago,
  PointsSectionVFDD_SparsDiagoSpars,
  PointsSectionVFDD_SparsDiagoSpaco,
  PointsSectionVFDD_SpacoDiagoDense,
  PointsSectionVFDD_SpacoDiagoDiago,
  PointsSectionVFDD_SpacoDiagoSpars,
  PointsSectionVFDD_SpacoDiagoSpaco
)
{

  testFunction<TestType, nelson::GaussNewton>();
  testFunction<TestType, nelson::LevenbergMarquardt>();

}

