#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mat/SparsityPattern.h"
#include "mat/DenseMatrixBlock.hpp"

#include "nelson/SingleSectionHessian.hpp"
#include "nelson/MatrixDenseWrapper.hpp"
#include "nelson/MatrixSparseWrapper.hpp"


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

TEMPLATE_TEST_CASE_SIG("DenseSparseWrapper", "[MatrixSparseDenseWrapper]",
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
  auto sp = std::make_shared< mat::SparsityPatternColMajor>(numBlocks, numBlocks);
  sp->setDiagonal();
  if (matType != mat::BlockDiagonal) {
    for (int i = 0; i < 10; i++) {
      int r = rand() % numBlocks;
      int c = rand() % numBlocks;
      sp->add(r, c);
    }
  }

  nelson::SingleSectionHessian<matType, double, B, NB> sec;
  sec.resize(SecType<B>::secSize, numBlocks, sp);
  sec.clearAll();
  REQUIRE(sec.chi2() == 0);

  {
    nelson::DenseWrapper<matType, double, mat::ColMajor, B, B, NB, NB> wrapperDense;
    wrapperDense.set(&sec.H());

    {
      const auto& wr = wrapperDense;
      auto mat2 = wr.mat() * 2;
    }

    nelson::SparseWrapper<matType, double, mat::ColMajor, B, B, NB, NB> wrapperSparse;
    wrapperSparse.set(&sec.H());

    {
      const auto& wr = wrapperSparse;
      auto mat2 = wr.mat() * 2;
    }
  }

  // dense square
  {
    nelson::DenseSquareWrapper<matType, double, mat::ColMajor, B, NB> wrapperSquareDense;
    wrapperSquareDense.set(&sec.H());

    nelson::DenseSquareWrapper<matType, double, mat::ColMajor, B, NB>::DiagType diagonal = wrapperSquareDense.diagonal();
    diagonal.setConstant(3);
    wrapperSquareDense.setDiagonal(diagonal);
    wrapperSquareDense.diagonalCopy(diagonal);
    REQUIRE((diagonal.array() == wrapperSquareDense.mat().diagonal().array()).all());

    wrapperSquareDense.mat().diagonal().setRandom();
    wrapperSquareDense.diagonalCopy(diagonal);
    REQUIRE((diagonal.array() == wrapperSquareDense.mat().diagonal().array()).all());

  }
  // sparse square
  {
    nelson::SparseSquareWrapper<matType, double, mat::ColMajor, B, NB> wrapperSquareSparse;
    wrapperSquareSparse.set(&sec.H());

    nelson::SparseSquareWrapper<matType, double, mat::ColMajor, B, NB>::DiagType diagonal = wrapperSquareSparse.diagonal();
    diagonal.setConstant(3);
    wrapperSquareSparse.setDiagonal(diagonal);
    wrapperSquareSparse.diagonalCopy(diagonal);
    REQUIRE((diagonal.array() == wrapperSquareSparse.mat().diagonal().array()).all());

    wrapperSquareSparse.mat().diagonal().setRandom();
    wrapperSquareSparse.diagonalCopy(diagonal);
    REQUIRE((diagonal.array() == wrapperSquareSparse.mat().diagonal().array()).all());
  }

}