#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mat/SparsityPattern.h"
#include "mat/DenseMatrixBlock.hpp"

#include "nelson/SingleSectionHessian.hpp"
#include "nelson/SingleSection.hpp"
#include "nelson/EdgeUnary.hpp"
#include "nelson/EdgeBinary.hpp"

#include <array>
#include <iostream>

struct Point {
  Eigen::Vector3d point;
};

static constexpr int secSizeFix = 3;
static constexpr int numBlocks = 10;

template<class Section>
class EdgeUnaryTest : public nelson::EdgeUnarySingleSectionCRPT<Section, EdgeUnaryTest<Section>> {
  int _parId;

public:
  EdgeUnaryTest(int parId) : _parId(parId) {}

  void update(const Point & point, bool hessians) {
    REQUIRE(this->parId() == _parId);
    REQUIRE(this->HUid() >= 0);
  }

  template<class Derived>
  void updateHBlock(Eigen::MatrixBase<Derived> & b) {
    std::cout << "EdgeUnaryTest::updateHBlock " << this->parId() << "," << this->parId() << std::endl;
    b.setConstant(1);
  }
};

template<class Section>
class EdgeBinaryTest : public nelson::EdgeBinarySingleSectionCRPT<Section, EdgeBinaryTest<Section>> {
  int _par1Id, _par2Id;

public:
  EdgeBinaryTest(int par1Id, int par2Id) : _par1Id(par1Id), _par2Id(par2Id) {}

  void update(const Point& point1, const Point& point2, bool hessians) {
    REQUIRE(this->par_1_Id() == _par1Id);
    REQUIRE(this->par_2_Id() == _par2Id);
    REQUIRE(this->H_11_Uid() >= 0);
    REQUIRE(this->H_12_Uid() >= 0);
    REQUIRE(this->H_22_Uid() >= 0);

  }

  template<class Derived>
  void updateH11Block(Eigen::MatrixBase<Derived> & b) {
    std::cout << "EdgeBinaryTest::updateHBlock[11] " << this->par_1_Id() << ","<< this->par_1_Id() << std::endl;
    b.setConstant(11);
  }
  template<class Derived>
  void updateH12Block(Eigen::MatrixBase<Derived> & b) {
    std::cout << "EdgeBinaryTest::updateHBlock[12] " << this->par_1_Id() << "," << this->par_2_Id() << std::endl;
    b.setConstant(12);
  }
  template<class Derived>
  void updateH22Block(Eigen::MatrixBase<Derived> & b) {
    std::cout << "EdgeBinaryTest::updateHBlock[22] " << this->par_2_Id() << "," << this->par_2_Id() << std::endl;
    b.setConstant(22);
  }

};

template<int matType>
class PointsSectionFF : public nelson::SingleSection<PointsSectionFF<matType>, Point, matType, double, secSizeFix, numBlocks> {
  std::array<Point, numBlocks> _points;
public:
  PointsSectionFF() {
    this->parametersReady();
  }

  virtual const Point& parameter(int i) const override {
    return _points[i];
  }
  virtual Point& parameter(int i) override {
    return _points[i];
  }

};

using PointsSectionFF_BlockDense = PointsSectionFF<mat::BlockDense>;
using PointsSectionFF_BlockDiagonal = PointsSectionFF<mat::BlockDiagonal>;
using PointsSectionFF_BlockSparse = PointsSectionFF<mat::BlockSparse>;
using PointsSectionFF_BlockCoeffSparse = PointsSectionFF<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matType>
class PointsSectionFD : public nelson::SingleSection<PointsSectionFD<matType>, Point, matType, double, secSizeFix, mat::Dynamic> {
  std::vector<Point> _points;
public:

  PointsSectionFD() {
    _points.resize(numBlocks);
    this->parametersReady();
  }

  virtual const Point& parameter(int i) const {
    return _points[i];
  }
  virtual Point& parameter(int i) {
    return _points[i];
  }

  int numParameters() const override {
    return _points.size();
  }

};

using PointsSectionFD_BlockDense = PointsSectionFD<mat::BlockDense>;
using PointsSectionFD_BlockDiagonal = PointsSectionFD<mat::BlockDiagonal>;
using PointsSectionFD_BlockSparse = PointsSectionFD<mat::BlockSparse>;
using PointsSectionFD_BlockCoeffSparse = PointsSectionFD<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matType>
class PointsSectionDF : public nelson::SingleSection<PointsSectionDF<matType>, Point, matType, double, mat::Dynamic, numBlocks> {
  std::array<Point, numBlocks> _points;
public:

  PointsSectionDF() {
    this->parametersReady();
  }

  virtual const Point& parameter(int i) const {
    return _points[i];
  }
  virtual Point& parameter(int i) {
    return _points[i];
  }

  int parameterSize() const override {
    return secSizeFix;
  }

};

using PointsSectionDF_BlockDense = PointsSectionDF<mat::BlockDense>;
using PointsSectionDF_BlockDiagonal = PointsSectionDF<mat::BlockDiagonal>;
using PointsSectionDF_BlockSparse = PointsSectionDF<mat::BlockSparse>;
using PointsSectionDF_BlockCoeffSparse = PointsSectionDF<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matType>
class PointsSectionDD : public nelson::SingleSection<PointsSectionDD<matType>, Point, matType, double, mat::Dynamic, mat::Dynamic> {
  std::vector<Point> _points;
public:

  PointsSectionDD() {
    _points.resize(numBlocks);
    this->parametersReady();
  }

  virtual const Point& parameter(int i) const {
    return _points[i];
  }
  virtual Point& parameter(int i) {
    return _points[i];
  }

  int parameterSize() const override {
    return secSizeFix;
  }

  int numParameters() const override {
    return _points.size();
  }

};

using PointsSectionDD_BlockDense = PointsSectionDD<mat::BlockDense>;
using PointsSectionDD_BlockDiagonal = PointsSectionDD<mat::BlockDiagonal>;
using PointsSectionDD_BlockSparse = PointsSectionDD<mat::BlockSparse>;
using PointsSectionDD_BlockCoeffSparse = PointsSectionDD<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matType>
class PointsSectionVF : public nelson::SingleSection<PointsSectionVF<matType>, Point, matType, double, mat::Variable, numBlocks> {
  std::array<Point, numBlocks> _points;
  std::vector<int> _sizes;
public:

  PointsSectionVF() : _sizes(numBlocks, secSizeFix)
  {
    this->parametersReady();
  }

  virtual const Point& parameter(int i) const {
    return _points[i];
  }
  virtual Point& parameter(int i) {
    return _points[i];
  }

  const std::vector<int>& parameterSize() const override {
    return _sizes;
  }

};

using PointsSectionVF_BlockDense = PointsSectionVF<mat::BlockDense>;
using PointsSectionVF_BlockDiagonal = PointsSectionVF<mat::BlockDiagonal>;
using PointsSectionVF_BlockSparse = PointsSectionVF<mat::BlockSparse>;
using PointsSectionVF_BlockCoeffSparse = PointsSectionVF<mat::BlockCoeffSparse>;

//-----------------------------------------------------------------------------------------------------------------

template<int matType>
class PointsSectionVD : public nelson::SingleSection<PointsSectionVD<matType>, Point, matType, double, mat::Variable, mat::Dynamic> {
  std::vector<Point> _points;
  std::vector<int> _sizes;
public:

  PointsSectionVD() : _sizes(numBlocks, secSizeFix) {
    _points.resize(numBlocks);
    this->parametersReady();
  }

  virtual const Point& parameter(int i) const {
    return _points[i];
  }
  virtual Point& parameter(int i) {
    return _points[i];
  }

  const std::vector<int>& parameterSize() const override {
    return _sizes;
  }

};

using PointsSectionVD_BlockDense = PointsSectionVD<mat::BlockDense>;
using PointsSectionVD_BlockDiagonal = PointsSectionVD<mat::BlockDiagonal>;
using PointsSectionVD_BlockSparse = PointsSectionVD<mat::BlockSparse>;
using PointsSectionVD_BlockCoeffSparse = PointsSectionVD<mat::BlockCoeffSparse>;


TEMPLATE_TEST_CASE("SingleSection", "[SingleSection]",
  PointsSectionFF_BlockDense, PointsSectionFF_BlockDiagonal, PointsSectionFF_BlockSparse, PointsSectionFF_BlockCoeffSparse,
  PointsSectionFD_BlockDense, PointsSectionFD_BlockDiagonal, PointsSectionFD_BlockSparse, PointsSectionFD_BlockCoeffSparse,
  PointsSectionDF_BlockDense, PointsSectionDF_BlockDiagonal, PointsSectionDF_BlockSparse, PointsSectionDF_BlockCoeffSparse,
  PointsSectionDD_BlockDense, PointsSectionDD_BlockDiagonal, PointsSectionDD_BlockSparse, PointsSectionDD_BlockCoeffSparse,
  PointsSectionVF_BlockDense, PointsSectionVF_BlockDiagonal, PointsSectionVF_BlockSparse, PointsSectionVF_BlockCoeffSparse,
  PointsSectionVD_BlockDense, PointsSectionVD_BlockDiagonal, PointsSectionVD_BlockSparse, PointsSectionVD_BlockCoeffSparse
)
{
  std::cout << "-------------------------------------------------------" << std::endl;
  TestType pss;
  //  REQUIRE(pss.parameterSize() == secSizeFix);
  REQUIRE(pss.numParameters() == numBlocks);

  // unary edge
  for (int i = 0; i < numBlocks; i++) {
    pss.addEdge(i, new EdgeUnaryTest<TestType>(i));
  }

  // add "odometry"  edges (not if mat diagonal)
  if (pss.matType() != mat::BlockDiagonal) {
    for (int i = 0; i < numBlocks - 1; i++) {
      pss.addEdge(i, i + 1, new EdgeBinaryTest<TestType>(i, i + 1));
    }
  }

  // add "extreme" bin relation (not if mat diagonal)
  if (pss.matType() != mat::BlockDiagonal) {
    for (int i = 0; i < numBlocks - 1; i++) {
      pss.addEdge(i, numBlocks - 1, new EdgeBinaryTest<TestType>(i, numBlocks - 1));
    }
  }


  // ready!
  pss.structureReady();

  for (int i = 0; i < numBlocks; i++) {
    REQUIRE(pss.sparsityPattern().has(i, i));
  }
  // check "odometry"  edges (not if mat diagonal)
  if (pss.matType() != mat::BlockDiagonal) {
    for (int i = 0; i < numBlocks; i++) {
      REQUIRE(pss.sparsityPattern().has(i, i));
    }
    for (int i = 0; i < numBlocks - 1; i++) {
      REQUIRE(pss.sparsityPattern().has(i, i + 1));
    }
  }

  // check "extreme" bin relation (not if mat diagonal)
  if (pss.matType() != mat::BlockDiagonal) {
    for (int i = 0; i < numBlocks - 1; i++) {
      REQUIRE(pss.sparsityPattern().has(i, numBlocks - 1));
    }
  }


  pss.update(true);

}



