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
class Points2d3dFDFD : public nelson::DoubleSection< Points2d3dFDFD<matTypeU, matTypeV, matTypeW>, Point2d, Point3d, matTypeU, matTypeV, matTypeW, double, Point2d::blockSize, mat::Dynamic, numPoints2d, mat::Dynamic> {
  std::array<Point2d, numPoints2d> _points2d;
  std::vector<Point3d> _points3d;
  Point2d _fixedPoint2d;
  Point3d _fixedPoint3d;
public:
  Points2d3dFDFD() {
    _points3d.resize(numPoints3d);
    this->parametersReady();
  }

  int parameterVSize() const override {
    return Point2d::blockSize;
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

using PointsSectionFDFD_DenseDenseDense = Points2d3dFDFD<mat::BlockDense, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFD_DenseDenseDiago = Points2d3dFDFD<mat::BlockDense, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFD_DenseDenseSpars = Points2d3dFDFD<mat::BlockDense, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFD_DenseDenseSpaco = Points2d3dFDFD<mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DenseDiagoDense = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_DenseDiagoDiago = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_DenseDiagoSpars = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_DenseDiagoSpaco = Points2d3dFDFD<mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DenseSparsDense = Points2d3dFDFD<mat::BlockDense, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFD_DenseSparsDiago = Points2d3dFDFD<mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_DenseSparsSpars = Points2d3dFDFD<mat::BlockDense, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFD_DenseSparsSpaco = Points2d3dFDFD<mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DenseSpacoDense = Points2d3dFDFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFD_DenseSpacoDiago = Points2d3dFDFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_DenseSpacoSpars = Points2d3dFDFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFD_DenseSpacoSpaco = Points2d3dFDFD<mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDFD_DiagoDenseDense = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFD_DiagoDenseDiago = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFD_DiagoDenseSpars = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFD_DiagoDenseSpaco = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DiagoDiagoDense = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_DiagoDiagoDiago = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_DiagoDiagoSpars = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_DiagoDiagoSpaco = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DiagoSparsDense = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFD_DiagoSparsDiago = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_DiagoSparsSpars = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFD_DiagoSparsSpaco = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFD_DiagoSpacoDense = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFD_DiagoSpacoDiago = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_DiagoSpacoSpars = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFD_DiagoSpacoSpaco = Points2d3dFDFD<mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDFD_SparsDenseDense = Points2d3dFDFD<mat::BlockSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFD_SparsDenseDiago = Points2d3dFDFD<mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFD_SparsDenseSpars = Points2d3dFDFD<mat::BlockSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFD_SparsDenseSpaco = Points2d3dFDFD<mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SparsDiagoDense = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_SparsDiagoDiago = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_SparsDiagoSpars = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_SparsDiagoSpaco = Points2d3dFDFD<mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SparsSparsDense = Points2d3dFDFD<mat::BlockSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFD_SparsSparsDiago = Points2d3dFDFD<mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_SparsSparsSpars = Points2d3dFDFD<mat::BlockSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFD_SparsSparsSpaco = Points2d3dFDFD<mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SparsSpacoDense = Points2d3dFDFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFD_SparsSpacoDiago = Points2d3dFDFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_SparsSpacoSpars = Points2d3dFDFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFD_SparsSpacoSpaco = Points2d3dFDFD<mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

using PointsSectionFDFD_SpacoDenseDense = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense>;
using PointsSectionFDFD_SpacoDenseDiago = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal>;
using PointsSectionFDFD_SpacoDenseSpars = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse>;
using PointsSectionFDFD_SpacoDenseSpaco = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SpacoDiagoDense = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense>;
using PointsSectionFDFD_SpacoDiagoDiago = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal>;
using PointsSectionFDFD_SpacoDiagoSpars = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse>;
using PointsSectionFDFD_SpacoDiagoSpaco = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SpacoSparsDense = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense>;
using PointsSectionFDFD_SpacoSparsDiago = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_SpacoSparsSpars = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse>;
using PointsSectionFDFD_SpacoSparsSpaco = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse>;
using PointsSectionFDFD_SpacoSpacoDense = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense>;
using PointsSectionFDFD_SpacoSpacoDiago = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal>;
using PointsSectionFDFD_SpacoSpacoSpars = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse>;
using PointsSectionFDFD_SpacoSpacoSpaco = Points2d3dFDFD<mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse>;

TEMPLATE_TEST_CASE("DoubleSection-FDFD", "[DoubleSection-FDFD]",
  PointsSectionFDFD_DenseDenseDense,
  PointsSectionFDFD_DenseDenseDiago,
  PointsSectionFDFD_DenseDenseSpars,
  PointsSectionFDFD_DenseDenseSpaco,
  PointsSectionFDFD_DenseDiagoDense,
  PointsSectionFDFD_DenseDiagoDiago,
  PointsSectionFDFD_DenseDiagoSpars,
  PointsSectionFDFD_DenseDiagoSpaco,
  PointsSectionFDFD_DenseSparsDense,
  PointsSectionFDFD_DenseSparsDiago,
  PointsSectionFDFD_DenseSparsSpars,
  PointsSectionFDFD_DenseSparsSpaco,
  PointsSectionFDFD_DenseSpacoDense,
  PointsSectionFDFD_DenseSpacoDiago,
  PointsSectionFDFD_DenseSpacoSpars,
  PointsSectionFDFD_DenseSpacoSpaco,
//------------------------------------
  PointsSectionFDFD_DiagoDenseDense,
  PointsSectionFDFD_DiagoDenseDiago,
  PointsSectionFDFD_DiagoDenseSpars,
  PointsSectionFDFD_DiagoDenseSpaco,
  PointsSectionFDFD_DiagoDiagoDense,
  PointsSectionFDFD_DiagoDiagoDiago,
  PointsSectionFDFD_DiagoDiagoSpars,
  PointsSectionFDFD_DiagoDiagoSpaco,
  PointsSectionFDFD_DiagoSparsDense,
  PointsSectionFDFD_DiagoSparsDiago,
  PointsSectionFDFD_DiagoSparsSpars,
  PointsSectionFDFD_DiagoSparsSpaco,
  PointsSectionFDFD_DiagoSpacoDense,
  PointsSectionFDFD_DiagoSpacoDiago,
  PointsSectionFDFD_DiagoSpacoSpars,
  PointsSectionFDFD_DiagoSpacoSpaco,
  //------------------------------------
  PointsSectionFDFD_SparsDenseDense,
  PointsSectionFDFD_SparsDenseDiago,
  PointsSectionFDFD_SparsDenseSpars,
  PointsSectionFDFD_SparsDenseSpaco,
  PointsSectionFDFD_SparsDiagoDense,
  PointsSectionFDFD_SparsDiagoDiago,
  PointsSectionFDFD_SparsDiagoSpars,
  PointsSectionFDFD_SparsDiagoSpaco,
  PointsSectionFDFD_SparsSparsDense,
  PointsSectionFDFD_SparsSparsDiago,
  PointsSectionFDFD_SparsSparsSpars,
  PointsSectionFDFD_SparsSparsSpaco,
  PointsSectionFDFD_SparsSpacoDense,
  PointsSectionFDFD_SparsSpacoDiago,
  PointsSectionFDFD_SparsSpacoSpars,
  PointsSectionFDFD_SparsSpacoSpaco,
  //------------------------------------
  PointsSectionFDFD_SpacoDenseDense,
  PointsSectionFDFD_SpacoDenseDiago,
  PointsSectionFDFD_SpacoDenseSpars,
  PointsSectionFDFD_SpacoDenseSpaco,
  PointsSectionFDFD_SpacoDiagoDense,
  PointsSectionFDFD_SpacoDiagoDiago,
  PointsSectionFDFD_SpacoDiagoSpars,
  PointsSectionFDFD_SpacoDiagoSpaco,
  PointsSectionFDFD_SpacoSparsDense,
  PointsSectionFDFD_SpacoSparsDiago,
  PointsSectionFDFD_SpacoSparsSpars,
  PointsSectionFDFD_SpacoSparsSpaco,
  PointsSectionFDFD_SpacoSpacoDense,
  PointsSectionFDFD_SpacoSpacoDiago,
  PointsSectionFDFD_SpacoSpacoSpars,
  PointsSectionFDFD_SpacoSpacoSpaco  
  )
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  REQUIRE(pss.numParametersU() == numPoints2d);
  REQUIRE(pss.numParametersV() == numPoints3d);


  pss.structureReady();
}