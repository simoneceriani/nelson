#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/MatrixWWtMultiplier.hpp"
#include "nelson/MatrixDenseWrapper.hpp"

#include <iostream>

constexpr int BRv = 6;
constexpr int NBRv = 3;

constexpr int BCv = 3;
constexpr int NBCv = 5;

#define DEBUGOUT if(false)

TEMPLATE_TEST_CASE_SIG("MatrixWWtMultiplier-Base", "[MatrixWWtMultiplier-Base]",
  ((int matType, int BR, int BC, int NBR, int NBC, int matOutType, int matOutOrdering), matType, BR, BC, NBR, NBC, matOutType, matOutOrdering),
  (mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse, mat::RowMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::RowMajor)
)
{

  typename mat::MatrixBlockIterableTypeTraits<matType, double, mat::RowMajor, BR, BC, NBR, NBC>::MatrixType W;

  mat::SparsityPattern<mat::RowMajor>::SPtr sp(new mat::SparsityPattern<mat::RowMajor>(NBRv, NBCv));
  // 0 1 2 3 4
  // X - - X X
  if (matType != mat::BlockDiagonal) {
    sp->add(0, 0); sp->add(0, 3); sp->add(0, 4);
    // X X - X -
    sp->add(1, 0); sp->add(1, 1); sp->add(1, 3);
    // - X X X X
    sp->add(2, 1); sp->add(2, 2); sp->add(2, 3); sp->add(2, 4);

    //   0 1 2 3 4       0 1 2
    // 0 X - - X X     0 X X - 
    // 1 X X - X -  *  1 - X X
    // 2 - X X X X     2 - - X
    //                 3 X X X
    //                 4 X - X

    // [0,0] = 0,0 * (0,0)' + 0,3 * (0,3)' + 0,4 * (0,4)'
    // [0,1] = 0,0 * (1,0)' + 0,3 * (1,3)'
    // [0,2] = 0,3 * (2,3)' + 0,4 * (2,4)'
    // etc
  }
  else {
    sp->setDiagonal();
  }

  W.resize(mat::MatrixBlockDescriptor<BR, BC, NBR, NBC>(BRv, BCv, NBRv, NBCv), sp);
  W.setZero();
  for (int bi = 0; bi < W.nonZeroBlocks(); bi++) {
    W.blockByUID(bi).setRandom();
  }

  nelson::MatrixWWtMultiplier<matType, double, BR, BC, NBR, NBC, matOutType, matOutOrdering> mwwt;
  mwwt.prepare(W);
  mwwt.multiply(W, W);

  nelson::DenseWrapper<matType, double, mat::RowMajor, BR, BC, NBR, NBC> W_Wrap;
  W_Wrap.set(&W);
  DEBUGOUT std::cout << "W_Wrap" << std::endl << W_Wrap.mat() << std::endl << std::endl;

  Eigen::MatrixXd WWt = W_Wrap.mat() * W_Wrap.mat().transpose();
  DEBUGOUT std::cout << "WWt" << std::endl << WWt << std::endl << std::endl;

  nelson::DenseWrapper<matOutType, double, matOutOrdering, BR, BR, NBR, NBR> WWt_Wrap;
  WWt_Wrap.set(&mwwt.result());

  DEBUGOUT std::cout << "WWt_Wrap" << std::endl << WWt_Wrap.mat() << std::endl << std::endl;

  Eigen::MatrixXd diff = (WWt_Wrap.mat() - WWt).triangularView<Eigen::Upper>();
  DEBUGOUT std::cout << "diff" << std::endl << diff << std::endl << std::endl;

  REQUIRE((diff.array().abs() < Eigen::NumTraits<double>::dummy_precision()).all());

}