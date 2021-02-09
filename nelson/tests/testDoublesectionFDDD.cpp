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
class Points2d3dFDDD : public nelson::DoubleSection< Points2d3dFDDD<matTypeU, matTypeV, matTypeW>, Point2d, Point3d, matTypeU, matTypeV, matTypeW, double, Point2d::blockSize, mat::Dynamic, mat::Dynamic, mat::Dynamic> {
  std::vector<Point2d> _points2d;
  std::vector<Point3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;
public:
  Points2d3dFDDD() {
    _points2d.resize(numPoints2d);
    _points3d.resize(numPoints3d);
    this->parametersReady();
  }

  int parameterVSize() const override {
    return Point2d::blockSize;
  }
  int numParametersU() const override {
    return _points2d.size();
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

using PointsSectionFDDD_DenseDenseDense = Points2d3dFDDD<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDD_DenseDenseDiago = Points2d3dFDDD<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDD_DenseDenseSpars = Points2d3dFDDD<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDD_DenseDenseSpaco = Points2d3dFDDD<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DenseDiagoDense = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_DenseDiagoDiago = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_DenseDiagoSpars = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_DenseDiagoSpaco = Points2d3dFDDD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DenseSparsDense = Points2d3dFDDD<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDD_DenseSparsDiago = Points2d3dFDDD<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_DenseSparsSpars = Points2d3dFDDD<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDD_DenseSparsSpaco = Points2d3dFDDD<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DenseSpacoDense = Points2d3dFDDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDD_DenseSpacoDiago = Points2d3dFDDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_DenseSpacoSpars = Points2d3dFDDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDD_DenseSpacoSpaco = Points2d3dFDDD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDDD_DiagoDenseDense = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDD_DiagoDenseDiago = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDD_DiagoDenseSpars = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDD_DiagoDenseSpaco = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DiagoDiagoDense = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_DiagoDiagoDiago = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_DiagoDiagoSpars = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_DiagoDiagoSpaco = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DiagoSparsDense = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDD_DiagoSparsDiago = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_DiagoSparsSpars = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDD_DiagoSparsSpaco = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDD_DiagoSpacoDense = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDD_DiagoSpacoDiago = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_DiagoSpacoSpars = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDD_DiagoSpacoSpaco = Points2d3dFDDD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDDD_SparsDenseDense = Points2d3dFDDD<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDD_SparsDenseDiago = Points2d3dFDDD<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDD_SparsDenseSpars = Points2d3dFDDD<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDD_SparsDenseSpaco = Points2d3dFDDD<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SparsDiagoDense = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_SparsDiagoDiago = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_SparsDiagoSpars = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_SparsDiagoSpaco = Points2d3dFDDD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SparsSparsDense = Points2d3dFDDD<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDD_SparsSparsDiago = Points2d3dFDDD<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_SparsSparsSpars = Points2d3dFDDD<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDD_SparsSparsSpaco = Points2d3dFDDD<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SparsSpacoDense = Points2d3dFDDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDD_SparsSpacoDiago = Points2d3dFDDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_SparsSpacoSpars = Points2d3dFDDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDD_SparsSpacoSpaco = Points2d3dFDDD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDDD_SpacoDenseDense = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDDD_SpacoDenseDiago = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDDD_SpacoDenseSpars = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDDD_SpacoDenseSpaco = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SpacoDiagoDense = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDDD_SpacoDiagoDiago = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDDD_SpacoDiagoSpars = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDDD_SpacoDiagoSpaco = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SpacoSparsDense = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDDD_SpacoSparsDiago = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_SpacoSparsSpars = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDDD_SpacoSparsSpaco = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDDD_SpacoSpacoDense = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDDD_SpacoSpacoDiago = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDDD_SpacoSpacoSpars = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDDD_SpacoSpacoSpaco = Points2d3dFDDD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FDDD", "[DoubleSection-FDDD]",
  PointsSectionFDDD_DenseDenseDense,
  PointsSectionFDDD_DenseDenseDiago,
  PointsSectionFDDD_DenseDenseSpars,
  PointsSectionFDDD_DenseDenseSpaco,
  PointsSectionFDDD_DenseDiagoDense,
  PointsSectionFDDD_DenseDiagoDiago,
  PointsSectionFDDD_DenseDiagoSpars,
  PointsSectionFDDD_DenseDiagoSpaco,
  PointsSectionFDDD_DenseSparsDense,
  PointsSectionFDDD_DenseSparsDiago,
  PointsSectionFDDD_DenseSparsSpars,
  PointsSectionFDDD_DenseSparsSpaco,
  PointsSectionFDDD_DenseSpacoDense,
  PointsSectionFDDD_DenseSpacoDiago,
  PointsSectionFDDD_DenseSpacoSpars,
  PointsSectionFDDD_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFDDD_DiagoDenseDense,
  PointsSectionFDDD_DiagoDenseDiago,
  PointsSectionFDDD_DiagoDenseSpars,
  PointsSectionFDDD_DiagoDenseSpaco,
  PointsSectionFDDD_DiagoDiagoDense,
  PointsSectionFDDD_DiagoDiagoDiago,
  PointsSectionFDDD_DiagoDiagoSpars,
  PointsSectionFDDD_DiagoDiagoSpaco,
  PointsSectionFDDD_DiagoSparsDense,
  PointsSectionFDDD_DiagoSparsDiago,
  PointsSectionFDDD_DiagoSparsSpars,
  PointsSectionFDDD_DiagoSparsSpaco,
  PointsSectionFDDD_DiagoSpacoDense,
  PointsSectionFDDD_DiagoSpacoDiago,
  PointsSectionFDDD_DiagoSpacoSpars,
  PointsSectionFDDD_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFDDD_SparsDenseDense,
  PointsSectionFDDD_SparsDenseDiago,
  PointsSectionFDDD_SparsDenseSpars,
  PointsSectionFDDD_SparsDenseSpaco,
  PointsSectionFDDD_SparsDiagoDense,
  PointsSectionFDDD_SparsDiagoDiago,
  PointsSectionFDDD_SparsDiagoSpars,
  PointsSectionFDDD_SparsDiagoSpaco,
  PointsSectionFDDD_SparsSparsDense,
  PointsSectionFDDD_SparsSparsDiago,
  PointsSectionFDDD_SparsSparsSpars,
  PointsSectionFDDD_SparsSparsSpaco,
  PointsSectionFDDD_SparsSpacoDense,
  PointsSectionFDDD_SparsSpacoDiago,
  PointsSectionFDDD_SparsSpacoSpars,
  PointsSectionFDDD_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFDDD_SpacoDenseDense,
  PointsSectionFDDD_SpacoDenseDiago,
  PointsSectionFDDD_SpacoDenseSpars,
  PointsSectionFDDD_SpacoDenseSpaco,
  PointsSectionFDDD_SpacoDiagoDense,
  PointsSectionFDDD_SpacoDiagoDiago,
  PointsSectionFDDD_SpacoDiagoSpars,
  PointsSectionFDDD_SpacoDiagoSpaco,
  PointsSectionFDDD_SpacoSparsDense,
  PointsSectionFDDD_SpacoSparsDiago,
  PointsSectionFDDD_SpacoSparsSpars,
  PointsSectionFDDD_SpacoSparsSpaco,
  PointsSectionFDDD_SpacoSpacoDense,
  PointsSectionFDDD_SpacoSpacoDiago,
  PointsSectionFDDD_SpacoSpacoSpars,
  PointsSectionFDDD_SpacoSpacoSpaco  
  )
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);

  pss.structureReady();

}