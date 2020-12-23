#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mat/SparsityPattern.h"
#include "mat/DenseMatrixBlock.hpp"

#include "nelson/SingleSectionHessian.hpp"


static constexpr int secSizeFix = 3;

template<int secType>
class  SecType {
public:
  static constexpr int secSize = secSizeFix;
};


template<>
class  SecType<mat::Variable> {
public:
  static const std::vector<int> secSize;
};
const std::vector<int> SecType<mat::Variable>::secSize = { 1, 2, 3, 4, 5 }; // sum = 15

static constexpr int numBlocks = 5;

TEMPLATE_TEST_CASE_SIG("SingleSectionHessian", "[SingleSectionHessian]",
  ((int matType, int B, int NB), matType, B, NB),
  (mat::BlockDense, secSizeFix, numBlocks),
  (mat::BlockDense, secSizeFix, mat::Dynamic),
  (mat::BlockDense, mat::Dynamic, numBlocks),
  (mat::BlockDense, mat::Dynamic, mat::Dynamic),
  (mat::BlockDense, mat::Variable, numBlocks),
  (mat::BlockDense, mat::Variable, mat::Dynamic),
  (mat::BlockDiagonal, secSizeFix, numBlocks),
  (mat::BlockDiagonal, secSizeFix, mat::Dynamic),
  (mat::BlockDiagonal, mat::Dynamic, numBlocks),
  (mat::BlockDiagonal, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::Variable, numBlocks),
  (mat::BlockDiagonal, mat::Variable, mat::Dynamic),
  (mat::BlockSparse, secSizeFix, numBlocks),
  (mat::BlockSparse, secSizeFix, mat::Dynamic),
  (mat::BlockSparse, mat::Dynamic, numBlocks),
  (mat::BlockSparse, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::Variable, numBlocks),
  (mat::BlockSparse, mat::Variable, mat::Dynamic),
  (mat::BlockCoeffSparse, secSizeFix, numBlocks),
  (mat::BlockCoeffSparse, secSizeFix, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::Dynamic, numBlocks),
  (mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::Variable, numBlocks),
  (mat::BlockCoeffSparse, mat::Variable, mat::Dynamic)
)
{
  mat::SparsityPatternColMajor sp(numBlocks, numBlocks);
  sp.setDiagonal();
  if (matType != mat::BlockDiagonal) {
    for (int i = 0; i < 10;i++) {
      int r = rand() % numBlocks;
      int c = rand() % numBlocks;
      sp.add(r, c);
    }
  }

  nelson::SingleSectionHessianEval<matType, double, B, NB> sec;
  sec.resize(SecType<B>::secSize, numBlocks, sp);
  sec.clearAll();
  sec.clearChi2();
  REQUIRE(sec.chi2() == 0);
}



TEST_CASE("temp", "temp") {
  // just compilation test

  mat::SparsityPatternColMajor sp(numBlocks, numBlocks);

  nelson::SingleSectionHessianEval<mat::BlockDense, double, secSizeFix, numBlocks>       sec_01;
  nelson::SingleSectionHessianEval<mat::BlockDense, double, secSizeFix, mat::Dynamic>    sec_02;
  nelson::SingleSectionHessianEval<mat::BlockDense, double, mat::Dynamic, secSizeFix>    sec_03;
  nelson::SingleSectionHessianEval<mat::BlockDense, double, mat::Dynamic, mat::Dynamic>  sec_04;
  nelson::SingleSectionHessianEval<mat::BlockDense, double, mat::Variable, secSizeFix>   sec_05;
  nelson::SingleSectionHessianEval<mat::BlockDense, double, mat::Variable, mat::Dynamic> sec_06;

  nelson::SingleSectionHessianEval<mat::BlockDiagonal, double, secSizeFix, numBlocks>       sec_11;
  nelson::SingleSectionHessianEval<mat::BlockDiagonal, double, secSizeFix, mat::Dynamic>    sec_12;
  nelson::SingleSectionHessianEval<mat::BlockDiagonal, double, mat::Dynamic, secSizeFix>    sec_13;
  nelson::SingleSectionHessianEval<mat::BlockDiagonal, double, mat::Dynamic, mat::Dynamic>  sec_14;
  nelson::SingleSectionHessianEval<mat::BlockDiagonal, double, mat::Variable, secSizeFix>   sec_15;
  nelson::SingleSectionHessianEval<mat::BlockDiagonal, double, mat::Variable, mat::Dynamic> sec_16;

  nelson::SingleSectionHessianEval<mat::BlockSparse, double, secSizeFix, numBlocks>       sec_21;
  nelson::SingleSectionHessianEval<mat::BlockSparse, double, secSizeFix, mat::Dynamic>    sec_22;
  nelson::SingleSectionHessianEval<mat::BlockSparse, double, mat::Dynamic, secSizeFix>    sec_23;
  nelson::SingleSectionHessianEval<mat::BlockSparse, double, mat::Dynamic, mat::Dynamic>  sec_24;
  nelson::SingleSectionHessianEval<mat::BlockSparse, double, mat::Variable, secSizeFix>   sec_25;
  nelson::SingleSectionHessianEval<mat::BlockSparse, double, mat::Variable, mat::Dynamic> sec_26;

  nelson::SingleSectionHessianEval<mat::BlockCoeffSparse, double, secSizeFix, numBlocks>       sec_31;
  nelson::SingleSectionHessianEval<mat::BlockCoeffSparse, double, secSizeFix, mat::Dynamic>    sec_32;
  nelson::SingleSectionHessianEval<mat::BlockCoeffSparse, double, mat::Dynamic, secSizeFix>    sec_33;
  nelson::SingleSectionHessianEval<mat::BlockCoeffSparse, double, mat::Dynamic, mat::Dynamic>  sec_34;
  nelson::SingleSectionHessianEval<mat::BlockCoeffSparse, double, mat::Variable, secSizeFix>   sec_35;
  nelson::SingleSectionHessianEval<mat::BlockCoeffSparse, double, mat::Variable, mat::Dynamic> sec_36;

}
