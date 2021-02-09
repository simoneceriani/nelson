#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/DoubleSectionHessian.hpp"
#include "nelson/DoubleSection.hpp"
#include "nelson/EdgeUnary.hpp"
#include "nelson/EdgeBinary.hpp"

#include <array>
#include <iostream>

struct Point2d {
  Eigen::Vector2d p2d;
  static constexpr int blockSize = 2;
};

struct Point3d {
  Eigen::Vector3d p3d;
  static constexpr int blockSize = 3;
};

static constexpr int numPoints2d = 5; // totsize = 10
static constexpr int numPoints3d = 3; // totsize = 9

template<int matTypeU, int matTypeV, int matTypeW>
class Points2d3dFFFD : public nelson::DoubleSection< Points2d3dFFFD<matTypeU, matTypeV, matTypeW>, Point2d, Point3d, matTypeU, matTypeV, matTypeW, double, Point2d::blockSize, Point3d::blockSize, numPoints2d, mat::Dynamic> {
  std::array<Point2d, numPoints2d> _points2d;
  std::vector<Point3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;
public:
  Points2d3dFFFD() {
    _points3d.resize(numPoints3d);
    this->parametersReady();
  }

  int numParametersV() const override {
    return _points3d.size();
  }

  virtual const Point2d& parameterU(nelson::NodeId i) const override {
    if (i.isVariable()) return _points2d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint2d;
    }
  }
  virtual Point2d& parameterU(nelson::NodeId i) override {
    if (i.isVariable()) return _points2d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint2d;
    }
  }

  virtual const Point3d& parameterV(nelson::NodeId i) const override {
    if (i.isVariable()) return _points3d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint3d;
    }
  }
  virtual Point3d& parameterV(nelson::NodeId i) override {
    if (i.isVariable()) return _points3d[i.id()];
    else {
      assert(i.id() == 0);
      return _fixedPoint3d;
    }
  }

  int numFixedParametersU() const override {
    return 1;
  }
  int numFixedParametersV() const override {
    return 1;
  }

};

using PointsSectionFFFD_DenseDenseDense = Points2d3dFFFD<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFD_DenseDenseDiago = Points2d3dFFFD<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFD_DenseDenseSpars = Points2d3dFFFD<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFD_DenseDenseSpaco = Points2d3dFFFD<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DenseDiagoDense = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_DenseDiagoDiago = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_DenseDiagoSpars = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_DenseDiagoSpaco = Points2d3dFFFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DenseSparsDense = Points2d3dFFFD<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFD_DenseSparsDiago = Points2d3dFFFD<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_DenseSparsSpars = Points2d3dFFFD<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFD_DenseSparsSpaco = Points2d3dFFFD<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DenseSpacoDense = Points2d3dFFFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFD_DenseSpacoDiago = Points2d3dFFFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_DenseSpacoSpars = Points2d3dFFFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFD_DenseSpacoSpaco = Points2d3dFFFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFFD_DiagoDenseDense = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFD_DiagoDenseDiago = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFD_DiagoDenseSpars = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFD_DiagoDenseSpaco = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DiagoDiagoDense = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_DiagoDiagoDiago = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_DiagoDiagoSpars = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_DiagoDiagoSpaco = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DiagoSparsDense = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFD_DiagoSparsDiago = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_DiagoSparsSpars = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFD_DiagoSparsSpaco = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFD_DiagoSpacoDense = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFD_DiagoSpacoDiago = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_DiagoSpacoSpars = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFD_DiagoSpacoSpaco = Points2d3dFFFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFFD_SparsDenseDense = Points2d3dFFFD<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFD_SparsDenseDiago = Points2d3dFFFD<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFD_SparsDenseSpars = Points2d3dFFFD<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFD_SparsDenseSpaco = Points2d3dFFFD<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SparsDiagoDense = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_SparsDiagoDiago = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_SparsDiagoSpars = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_SparsDiagoSpaco = Points2d3dFFFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SparsSparsDense = Points2d3dFFFD<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFD_SparsSparsDiago = Points2d3dFFFD<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_SparsSparsSpars = Points2d3dFFFD<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFD_SparsSparsSpaco = Points2d3dFFFD<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SparsSpacoDense = Points2d3dFFFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFD_SparsSpacoDiago = Points2d3dFFFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_SparsSpacoSpars = Points2d3dFFFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFD_SparsSpacoSpaco = Points2d3dFFFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFFFD_SpacoDenseDense = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFFFD_SpacoDenseDiago = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFFFD_SpacoDenseSpars = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFFFD_SpacoDenseSpaco = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SpacoDiagoDense = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFFFD_SpacoDiagoDiago = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFFFD_SpacoDiagoSpars = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFFFD_SpacoDiagoSpaco = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SpacoSparsDense = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFFFD_SpacoSparsDiago = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_SpacoSparsSpars = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFFFD_SpacoSparsSpaco = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFFFD_SpacoSpacoDense = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFFFD_SpacoSpacoDiago = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFFFD_SpacoSpacoSpars = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFFFD_SpacoSpacoSpaco = Points2d3dFFFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FFFD", "[DoubleSection-FFFD]",
  PointsSectionFFFD_DenseDenseDense,
  PointsSectionFFFD_DenseDenseDiago,
  PointsSectionFFFD_DenseDenseSpars,
  PointsSectionFFFD_DenseDenseSpaco,
  PointsSectionFFFD_DenseDiagoDense,
  PointsSectionFFFD_DenseDiagoDiago,
  PointsSectionFFFD_DenseDiagoSpars,
  PointsSectionFFFD_DenseDiagoSpaco,
  PointsSectionFFFD_DenseSparsDense,
  PointsSectionFFFD_DenseSparsDiago,
  PointsSectionFFFD_DenseSparsSpars,
  PointsSectionFFFD_DenseSparsSpaco,
  PointsSectionFFFD_DenseSpacoDense,
  PointsSectionFFFD_DenseSpacoDiago,
  PointsSectionFFFD_DenseSpacoSpars,
  PointsSectionFFFD_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFFFD_DiagoDenseDense,
  PointsSectionFFFD_DiagoDenseDiago,
  PointsSectionFFFD_DiagoDenseSpars,
  PointsSectionFFFD_DiagoDenseSpaco,
  PointsSectionFFFD_DiagoDiagoDense,
  PointsSectionFFFD_DiagoDiagoDiago,
  PointsSectionFFFD_DiagoDiagoSpars,
  PointsSectionFFFD_DiagoDiagoSpaco,
  PointsSectionFFFD_DiagoSparsDense,
  PointsSectionFFFD_DiagoSparsDiago,
  PointsSectionFFFD_DiagoSparsSpars,
  PointsSectionFFFD_DiagoSparsSpaco,
  PointsSectionFFFD_DiagoSpacoDense,
  PointsSectionFFFD_DiagoSpacoDiago,
  PointsSectionFFFD_DiagoSpacoSpars,
  PointsSectionFFFD_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFFFD_SparsDenseDense,
  PointsSectionFFFD_SparsDenseDiago,
  PointsSectionFFFD_SparsDenseSpars,
  PointsSectionFFFD_SparsDenseSpaco,
  PointsSectionFFFD_SparsDiagoDense,
  PointsSectionFFFD_SparsDiagoDiago,
  PointsSectionFFFD_SparsDiagoSpars,
  PointsSectionFFFD_SparsDiagoSpaco,
  PointsSectionFFFD_SparsSparsDense,
  PointsSectionFFFD_SparsSparsDiago,
  PointsSectionFFFD_SparsSparsSpars,
  PointsSectionFFFD_SparsSparsSpaco,
  PointsSectionFFFD_SparsSpacoDense,
  PointsSectionFFFD_SparsSpacoDiago,
  PointsSectionFFFD_SparsSpacoSpars,
  PointsSectionFFFD_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFFFD_SpacoDenseDense,
  PointsSectionFFFD_SpacoDenseDiago,
  PointsSectionFFFD_SpacoDenseSpars,
  PointsSectionFFFD_SpacoDenseSpaco,
  PointsSectionFFFD_SpacoDiagoDense,
  PointsSectionFFFD_SpacoDiagoDiago,
  PointsSectionFFFD_SpacoDiagoSpars,
  PointsSectionFFFD_SpacoDiagoSpaco,
  PointsSectionFFFD_SpacoSparsDense,
  PointsSectionFFFD_SpacoSparsDiago,
  PointsSectionFFFD_SpacoSparsSpars,
  PointsSectionFFFD_SpacoSparsSpaco,
  PointsSectionFFFD_SpacoSpacoDense,
  PointsSectionFFFD_SpacoSpacoDiago,
  PointsSectionFFFD_SpacoSpacoSpars,
  PointsSectionFFFD_SpacoSpacoSpaco  
  )
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);



}