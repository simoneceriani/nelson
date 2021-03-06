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

template<int matUType, int matType, int BR, int BC, int NBR, int NBC, int matOutType>
void templateTestFunc() {
  typename mat::MatrixBlockIterableTypeTraits<matUType, double, mat::ColMajor, BR, BR, NBR, NBR>::MatrixType U;
  mat::SparsityPattern<mat::ColMajor>::SPtr spU(new mat::SparsityPattern<mat::ColMajor>(NBRv, NBRv));
  spU->setDiagonal();
  if (matUType != mat::BlockDiagonal) {
    spU->add(0, 2);
  }

  U.resize(mat::MatrixBlockDescriptor<BR, BR, NBR, NBR>(BRv, NBRv, BRv, NBRv), spU);
  U.setZero();
  for (int bi = 0; bi < U.nonZeroBlocks(); bi++) {
    U.blockByUID(bi).setRandom();
  }

  typename mat::MatrixBlockIterableTypeTraits<matType, double, mat::RowMajor, BR, BC, NBR, NBC>::MatrixType W;

  mat::SparsityPattern<mat::RowMajor>::SPtr sp(new mat::SparsityPattern<mat::RowMajor>(NBRv, NBCv));
  if (matType != mat::BlockDiagonal) {
    // 0 1 2 3 4
    // X - - X X
    sp->add(0, 0); sp->add(0, 3); sp->add(0, 4);
    // X X - X -
    sp->add(1, 0); sp->add(1, 1); sp->add(1, 3);
    // - X X X X
    sp->add(2, 1); sp->add(2, 2); //sp->add(2, 3); sp->add(2, 4);

    //   0 1 2 3 4       0 1 2
    // 0 X - - X X     0 X X - 
    // 1 X X - X -  *  1 - X X
    // 2 - X X - -     2 - - X
    //                 3 X X -
    //                 4 X - -

    // [0,0] = 0,0 * (0,0)' + 0,3 * (0,3)' + 0,4 * (0,4)'
    // [0,1] = 0,0 * (1,0)' + 0,3 * (1,3)'
    // [0,2] = 0,3 * (2,3)' + 0,4 * (2,4)'
    // etc

    // result patter (tri up)
    // X X -
    //   X X
    //     X
  }
  else {
    sp->setDiagonal();
  }

  W.resize(mat::MatrixBlockDescriptor<BR, BC, NBR, NBC>(BRv, BCv, NBRv, NBCv), sp);
  W.setZero();
  for (int bi = 0; bi < W.nonZeroBlocks(); bi++) {
    W.blockByUID(bi).setRandom();
  }

  nelson::MatrixWWtMultiplier<matOutType, double, BR, NBR> mwwt;

  SECTION("SINGLE THREAD") {
    mwwt.settings().setSingleThread();
  }
  SECTION("MULTI THREAD") {
    mwwt.settings().setNumThreadsMax();
  }

  mwwt.prepare(U, W);
  mwwt.multiply(U, W, W);

  nelson::DenseWrapper<matType, double, mat::RowMajor, BR, BC, NBR, NBC> W_Wrap;
  W_Wrap.set(&W);
  DEBUGOUT std::cout << "W_Wrap" << std::endl << W_Wrap.mat() << std::endl << std::endl;
  nelson::DenseWrapper<matUType, double, mat::ColMajor, BR, BR, NBR, NBR> U_Wrap;
  U_Wrap.set(&U);
  DEBUGOUT std::cout << "U_Wrap" << std::endl << U_Wrap.mat() << std::endl << std::endl;

  Eigen::MatrixXd WWt = U_Wrap.mat() - W_Wrap.mat() * W_Wrap.mat().transpose();
  DEBUGOUT std::cout << "WWt" << std::endl << WWt << std::endl << std::endl;

  nelson::DenseWrapper<matOutType, double, mat::ColMajor, BR, BR, NBR, NBR> WWt_Wrap;
  WWt_Wrap.set(&mwwt.result());

  DEBUGOUT std::cout << "WWt_Wrap" << std::endl << WWt_Wrap.mat() << std::endl << std::endl;

  Eigen::MatrixXd diff = (WWt_Wrap.mat() - WWt).triangularView<Eigen::Upper>();
  DEBUGOUT std::cout << "diff" << std::endl << diff << std::endl << std::endl;

  bool ok = (diff.array().abs() < Eigen::NumTraits<double>::dummy_precision()).all();
  REQUIRE(ok);

}

TEMPLATE_TEST_CASE_SIG("MatrixWWtMultiplier-Dense", "[MatrixWWtMultiplier-Base]",
  ((int matUType, int matType, int BR, int BC, int NBR, int NBC, int matOutType), matUType, matType, BR, BC, NBR, NBC, matOutType),
  //***************************************************************************
  (mat::BlockDense, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDense, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse)
)
{
  templateTestFunc< matUType, matType, BR, BC, NBR, NBC, matOutType>();
}

TEMPLATE_TEST_CASE_SIG("MatrixWWtMultiplier-Sparse", "[MatrixWWtMultiplier-Base]",
  ((int matUType, int matType, int BR, int BC, int NBR, int NBC, int matOutType), matUType, matType, BR, BC, NBR, NBC, matOutType),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse)
)
{
  templateTestFunc< matUType, matType, BR, BC, NBR, NBC, matOutType>();
}


TEMPLATE_TEST_CASE_SIG("MatrixWWtMultiplier-SparseCoeff", "[MatrixWWtMultiplier-Base]",
  ((int matUType, int matType, int BR, int BC, int NBR, int NBC, int matOutType), matUType, matType, BR, BC, NBR, NBC, matOutType),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockCoeffSparse, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse)
)
{
  templateTestFunc< matUType, matType, BR, BC, NBR, NBC, matOutType>();
}
   
TEMPLATE_TEST_CASE_SIG("MatrixWWtMultiplier-Diagonal", "[MatrixWWtMultiplier-Base]",
  ((int matUType, int matType, int BR, int BC, int NBR, int NBC, int matOutType), matUType, matType, BR, BC, NBR, NBC, matOutType),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, NBRv, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse),
  //---------------------------------------------------------------------------------
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockDense),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockSparse),
  (mat::BlockDiagonal, mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse) 
)
{
  templateTestFunc< matUType, matType, BR, BC, NBR, NBC, matOutType>();
}
