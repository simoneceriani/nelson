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
class Points2d3dFDFF : public nelson::DoubleSection< Points2d3dFDFF<matTypeU, matTypeV, matTypeW>, Point2d, Point3d, matTypeU, matTypeV, matTypeW, double, Point2d::blockSize, mat::Dynamic, numPoints2d, numPoints3d> {
  std::array<Point2d, numPoints2d> _points2d;
  std::array<Point3d, numPoints3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;
public:
  Points2d3dFDFF() {
    this->parametersReady();
  }

  int parameterVSize() const override {
    return Point2d::blockSize;
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

using PointsSectionFDFF_DenseDenseDense = Points2d3dFDFF<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFF_DenseDenseDiago = Points2d3dFDFF<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFF_DenseDenseSpars = Points2d3dFDFF<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFF_DenseDenseSpaco = Points2d3dFDFF<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFF_DenseDiagoDense = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_DenseDiagoDiago = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_DenseDiagoSpars = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_DenseDiagoSpaco = Points2d3dFDFF<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFF_DenseSparsDense = Points2d3dFDFF<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFF_DenseSparsDiago = Points2d3dFDFF<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_DenseSparsSpars = Points2d3dFDFF<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFF_DenseSparsSpaco = Points2d3dFDFF<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFF_DenseSpacoDense = Points2d3dFDFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFF_DenseSpacoDiago = Points2d3dFDFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_DenseSpacoSpars = Points2d3dFDFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFF_DenseSpacoSpaco = Points2d3dFDFF<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDFF_DiagoDenseDense = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFF_DiagoDenseDiago = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFF_DiagoDenseSpars = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFF_DiagoDenseSpaco = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFF_DiagoDiagoDense = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_DiagoDiagoDiago = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_DiagoDiagoSpars = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_DiagoDiagoSpaco = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFF_DiagoSparsDense = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFF_DiagoSparsDiago = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_DiagoSparsSpars = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFF_DiagoSparsSpaco = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFF_DiagoSpacoDense = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFF_DiagoSpacoDiago = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_DiagoSpacoSpars = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFF_DiagoSpacoSpaco = Points2d3dFDFF<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDFF_SparsDenseDense = Points2d3dFDFF<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFF_SparsDenseDiago = Points2d3dFDFF<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFF_SparsDenseSpars = Points2d3dFDFF<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFF_SparsDenseSpaco = Points2d3dFDFF<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SparsDiagoDense = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_SparsDiagoDiago = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_SparsDiagoSpars = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_SparsDiagoSpaco = Points2d3dFDFF<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SparsSparsDense = Points2d3dFDFF<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFF_SparsSparsDiago = Points2d3dFDFF<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_SparsSparsSpars = Points2d3dFDFF<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFF_SparsSparsSpaco = Points2d3dFDFF<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SparsSpacoDense = Points2d3dFDFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFF_SparsSpacoDiago = Points2d3dFDFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_SparsSpacoSpars = Points2d3dFDFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFF_SparsSpacoSpaco = Points2d3dFDFF<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDFF_SpacoDenseDense = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFF_SpacoDenseDiago = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFF_SpacoDenseSpars = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFF_SpacoDenseSpaco = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SpacoDiagoDense = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFF_SpacoDiagoDiago = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFF_SpacoDiagoSpars = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFF_SpacoDiagoSpaco = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SpacoSparsDense = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFF_SpacoSparsDiago = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_SpacoSparsSpars = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFF_SpacoSparsSpaco = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFF_SpacoSpacoDense = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFF_SpacoSpacoDiago = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFF_SpacoSpacoSpars = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFF_SpacoSpacoSpaco = Points2d3dFDFF<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FDFF", "[DoubleSection-FDFF]",
  PointsSectionFDFF_DenseDenseDense,
  PointsSectionFDFF_DenseDenseDiago,
  PointsSectionFDFF_DenseDenseSpars,
  PointsSectionFDFF_DenseDenseSpaco,
  PointsSectionFDFF_DenseDiagoDense,
  PointsSectionFDFF_DenseDiagoDiago,
  PointsSectionFDFF_DenseDiagoSpars,
  PointsSectionFDFF_DenseDiagoSpaco,
  PointsSectionFDFF_DenseSparsDense,
  PointsSectionFDFF_DenseSparsDiago,
  PointsSectionFDFF_DenseSparsSpars,
  PointsSectionFDFF_DenseSparsSpaco,
  PointsSectionFDFF_DenseSpacoDense,
  PointsSectionFDFF_DenseSpacoDiago,
  PointsSectionFDFF_DenseSpacoSpars,
  PointsSectionFDFF_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFDFF_DiagoDenseDense,
  PointsSectionFDFF_DiagoDenseDiago,
  PointsSectionFDFF_DiagoDenseSpars,
  PointsSectionFDFF_DiagoDenseSpaco,
  PointsSectionFDFF_DiagoDiagoDense,
  PointsSectionFDFF_DiagoDiagoDiago,
  PointsSectionFDFF_DiagoDiagoSpars,
  PointsSectionFDFF_DiagoDiagoSpaco,
  PointsSectionFDFF_DiagoSparsDense,
  PointsSectionFDFF_DiagoSparsDiago,
  PointsSectionFDFF_DiagoSparsSpars,
  PointsSectionFDFF_DiagoSparsSpaco,
  PointsSectionFDFF_DiagoSpacoDense,
  PointsSectionFDFF_DiagoSpacoDiago,
  PointsSectionFDFF_DiagoSpacoSpars,
  PointsSectionFDFF_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFDFF_SparsDenseDense,
  PointsSectionFDFF_SparsDenseDiago,
  PointsSectionFDFF_SparsDenseSpars,
  PointsSectionFDFF_SparsDenseSpaco,
  PointsSectionFDFF_SparsDiagoDense,
  PointsSectionFDFF_SparsDiagoDiago,
  PointsSectionFDFF_SparsDiagoSpars,
  PointsSectionFDFF_SparsDiagoSpaco,
  PointsSectionFDFF_SparsSparsDense,
  PointsSectionFDFF_SparsSparsDiago,
  PointsSectionFDFF_SparsSparsSpars,
  PointsSectionFDFF_SparsSparsSpaco,
  PointsSectionFDFF_SparsSpacoDense,
  PointsSectionFDFF_SparsSpacoDiago,
  PointsSectionFDFF_SparsSpacoSpars,
  PointsSectionFDFF_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFDFF_SpacoDenseDense,
  PointsSectionFDFF_SpacoDenseDiago,
  PointsSectionFDFF_SpacoDenseSpars,
  PointsSectionFDFF_SpacoDenseSpaco,
  PointsSectionFDFF_SpacoDiagoDense,
  PointsSectionFDFF_SpacoDiagoDiago,
  PointsSectionFDFF_SpacoDiagoSpars,
  PointsSectionFDFF_SpacoDiagoSpaco,
  PointsSectionFDFF_SpacoSparsDense,
  PointsSectionFDFF_SpacoSparsDiago,
  PointsSectionFDFF_SpacoSparsSpars,
  PointsSectionFDFF_SpacoSparsSpaco,
  PointsSectionFDFF_SpacoSpacoDense,
  PointsSectionFDFF_SpacoSpacoDiago,
  PointsSectionFDFF_SpacoSpacoSpars,
  PointsSectionFDFF_SpacoSpacoSpaco  
  )
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);


  pss.structureReady();
}