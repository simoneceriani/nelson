#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mat/SparsityPattern.h"
#include "mat/DenseMatrixBlock.hpp"

#include "nelson/DoubleSectionHessian.hpp"

static constexpr int secUSizeFix = 6;
static constexpr int secVSizeFix = 3;

template<int secUType, int secVType>
class  SecType {
public:
  static constexpr int secUSize = secUSizeFix;
  static constexpr int secVSize = secVSizeFix;
};


template<int secVType>
class  SecType<mat::Variable, secVType> {
public:
  static const std::vector<int> secUSize;
  static constexpr int secVSize = secVSizeFix;
};
template<int secVType>
const std::vector<int> SecType<mat::Variable, secVType>::secUSize = { 6,5,7,4,8 }; // sum = 30

template<int secUType>
class  SecType<secUType, mat::Variable> {
public:
  static constexpr int secUSize = secUSizeFix;
  static const std::vector<int> secVSize;
};
template<int secUType>
const std::vector<int> SecType<secUType, mat::Variable>::secVSize = { 3,1,4 }; // sum = 9

template<>
class  SecType<mat::Variable, mat::Variable> {
public:
  static const std::vector<int> secUSize;
  static const std::vector<int> secVSize;
};
const std::vector<int> SecType<mat::Variable, mat::Variable>::secUSize = { 6,5,7,4,8 }; // sum = 30
const std::vector<int> SecType<mat::Variable, mat::Variable>::secVSize = { 3,1,4 }; // sum = 9

static constexpr int numUBlocks = 5;
static constexpr int numVBlocks = 3;

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
) {

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}


TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDense, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{
  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);
}

//*****************************************************************************************************************************
//*****************************************************************************************************************************
//*****************************************************************************************************************************

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
) {

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}


TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{
  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);
}

//*****************************************************************************************************************************
//*****************************************************************************************************************************
//*****************************************************************************************************************************

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
) {

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}


TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{
  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);
}


//*****************************************************************************************************************************
//*****************************************************************************************************************************
//*****************************************************************************************************************************

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDense, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
) {

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}


TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{

  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

}

TEMPLATE_TEST_CASE_SIG("DoubleSectionHessian", "[DoubleSectionHessian]",
  ((int matTypeU, int matTypeV, int matTypeW, int BU, int BV, int NBU, int NBV), matTypeU, matTypeV, matTypeW, BU, BV, NBU, NBV),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDense, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockDiagonal, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic),
  //-------------------------------------------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, secUSizeFix, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Dynamic, mat::Variable, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, secVSizeFix, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, numUBlocks, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, numVBlocks),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::BlockCoeffSparse, mat::Variable, mat::Variable, mat::Dynamic, mat::Dynamic)
)
{
  auto spU = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numUBlocks);
  spU->setDiagonal();
  if (matTypeU != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numUBlocks;
      spU->add(r, c);
    }
  }

  auto spV = std::make_shared<mat::SparsityPatternColMajor>(numVBlocks, numVBlocks);
  spV->setDiagonal();
  if (matTypeV != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numVBlocks;
      int c = rand() % numVBlocks;
      spV->add(r, c);
    }
  }

  auto spW = std::make_shared<mat::SparsityPatternColMajor>(numUBlocks, numVBlocks);
  spW->setDiagonal();
  if (matTypeW != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numUBlocks;
      int c = rand() % numVBlocks;
      spW->add(r, c);
    }
  }

  nelson::DoubleSectionHessian<matTypeU, matTypeV, matTypeW, double, BU, BV, NBU, NBV> sec;
  sec.resize(SecType<BU, BV>::secUSize, numUBlocks, SecType<BU, BV>::secVSize, numVBlocks, spU, spV, spW);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);
}
