#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/MatrixWVinvMultiplier.hpp"
#include "nelson/MatrixDenseWrapper.hpp"

#include <iostream>

constexpr int BRv = 6;
constexpr int NBRv = 3;

constexpr int BCv = 3;
constexpr int NBCv = 5;

#define DEBUGOUT if(false)

template<int matType, int BR, int BC, int NBR, int NBC>
void templateTestFunc() {
  typename mat::MatrixBlockIterableTypeTraits<mat::BlockDiagonal, double, mat::ColMajor, BC, BC, NBC, NBC>::MatrixType Vinv;
  mat::SparsityPattern<mat::ColMajor>::SPtr spVinv(new mat::SparsityPattern<mat::ColMajor>(NBCv, NBCv));
  spVinv->setDiagonal();

  auto blockDescriptorV = mat::MatrixBlockDescriptor<BC, BC, NBC, NBC>(BCv, NBCv, BCv, NBCv);
  auto blockDescriptorW = mat::MatrixBlockDescriptor<BR, BC, NBR, NBC>(BRv, BCv, NBRv, NBCv);

  Vinv.resize(blockDescriptorV, spVinv);
  Vinv.setZero();
  for (int bi = 0; bi < Vinv.nonZeroBlocks(); bi++) {
    Vinv.blockByUID(bi).setRandom();
  }

  mat::VectorBlock<double, BC, NBC> b;
  b.resize(blockDescriptorV.rowDescriptionCSPtr());
  b.mat().setRandom();

  mat::VectorBlock<double, BR, NBR> bv, bvOrig;
  bv.resize(blockDescriptorW.rowDescriptionCSPtr());
  bvOrig.resize(blockDescriptorW.rowDescriptionCSPtr());
  bvOrig.mat().setRandom();
  bv.mat() = bvOrig.mat();

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

    //   0 1 2 3 4  
    // 0 X - - X X   
    // 1 X X - X -  
    // 2 - X X - -  
    //              
    //              

  }
  else {
    sp->setDiagonal();
  }

  W.resize(blockDescriptorW, sp);
  W.setZero();
  for (int bi = 0; bi < W.nonZeroBlocks(); bi++) {
    W.blockByUID(bi).setRandom();
  }

  nelson::MatrixWVinvMultiplier<matType, double, BR, BC, NBR, NBC> mwvinv;

  SECTION("SINGLE THREAD") {
    mwvinv.settings().multiplication.setSingleThread();
    mwvinv.settings().rightVectorMult.setSingleThread();
  }
  SECTION("MULTI THREAD") {
    mwvinv.settings().multiplication.setNumThreadsMax();
    mwvinv.settings().rightVectorMult.setSingleThread();
  }

  mwvinv.prepare(W);
  mwvinv.multiply(W, Vinv);
  mwvinv.rightMultVectorSub(b, bv);

  nelson::DenseWrapper<matType, double, mat::RowMajor, BR, BC, NBR, NBC> W_Wrap;

  W_Wrap.set(&W);
  DEBUGOUT std::cout << "W_Wrap" << std::endl << W_Wrap.mat() << std::endl << std::endl;
  nelson::DenseWrapper<mat::BlockDiagonal, double, mat::ColMajor, BC, BC, NBC, NBC> Vinv_Wrap;
  Vinv_Wrap.set(&Vinv);
  DEBUGOUT std::cout << "Vinv_Wrap" << std::endl << Vinv_Wrap.mat() << std::endl << std::endl;

  Eigen::MatrixXd WVinv = W_Wrap.mat() * Vinv_Wrap.mat();
  DEBUGOUT std::cout << "WVinv" << std::endl << WVinv << std::endl << std::endl;

  nelson::DenseWrapper<matType, double, mat::RowMajor, BR, BC, NBR, NBC> WVinv_Wrap;
  WVinv_Wrap.set(&mwvinv.result());

  DEBUGOUT std::cout << "WVinv_Wrap" << std::endl << WVinv_Wrap.mat() << std::endl << std::endl;

  Eigen::MatrixXd diff = (WVinv_Wrap.mat() - WVinv).template triangularView<Eigen::Upper>();
  DEBUGOUT std::cout << "diff" << std::endl << diff << std::endl << std::endl;

  bool ok = (diff.array().abs() < Eigen::NumTraits<double>::dummy_precision()).all();
  REQUIRE(ok);

  Eigen::VectorXd diff2 = (bvOrig.mat() - WVinv_Wrap.mat() * b.mat()) - bv.mat();
  DEBUGOUT std::cout << "diff" << std::endl << diff.transpose() << std::endl << std::endl;

  bool ok2 = (diff2.array().abs() < Eigen::NumTraits<double>::dummy_precision()).all();
  REQUIRE(ok2);


}

TEMPLATE_TEST_CASE_SIG("MatrixWVinvMultiplier-Dense", "[MatrixWVinvMultiplier-Base]",
  ((int matType, int BR, int BC, int NBR, int NBC), matType, BR, BC, NBR, NBC),
  (mat::BlockDense, BRv, BCv, NBRv, NBCv),
  (mat::BlockSparse, BRv, BCv, NBRv, NBCv),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, NBCv),
  (mat::BlockDiagonal, BRv, BCv, NBRv, NBCv),
  //----------------------------------------
  (mat::BlockDense, BRv, BCv, NBRv, mat::Dynamic),
  (mat::BlockSparse, BRv, BCv, NBRv, mat::Dynamic),
  (mat::BlockCoeffSparse, BRv, BCv, NBRv, mat::Dynamic),
  (mat::BlockDiagonal, BRv, BCv, NBRv, mat::Dynamic),
  //----------------------------------------
  (mat::BlockDense, BRv, BCv, mat::Dynamic, NBCv),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, NBCv),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, NBCv),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, NBCv),
  //----------------------------------------
  (mat::BlockDense, BRv, BCv, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, BRv, BCv, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, BRv, BCv, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, BRv, BCv, mat::Dynamic, mat::Dynamic),
  //***************************************************************************
  (mat::BlockDense, BRv, mat::Dynamic, NBRv, NBCv),
  (mat::BlockSparse, BRv, mat::Dynamic, NBRv, NBCv),
  (mat::BlockCoeffSparse, BRv, mat::Dynamic, NBRv, NBCv),
  (mat::BlockDiagonal, BRv, mat::Dynamic, NBRv, NBCv),
  //----------------------------------------
  (mat::BlockDense, BRv, mat::Dynamic, NBRv, mat::Dynamic),
  (mat::BlockSparse, BRv, mat::Dynamic, NBRv, mat::Dynamic),
  (mat::BlockCoeffSparse, BRv, mat::Dynamic, NBRv, mat::Dynamic),
  (mat::BlockDiagonal, BRv, mat::Dynamic, NBRv, mat::Dynamic),
  //----------------------------------------
  (mat::BlockDense, BRv, mat::Dynamic, mat::Dynamic, NBCv),
  (mat::BlockSparse, BRv, mat::Dynamic, mat::Dynamic, NBCv),
  (mat::BlockCoeffSparse, BRv, mat::Dynamic, mat::Dynamic, NBCv),
  (mat::BlockDiagonal, BRv, mat::Dynamic, mat::Dynamic, NBCv),
  //----------------------------------------
  (mat::BlockDense, BRv, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, BRv, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, BRv, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, BRv, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  //***************************************************************************
  //***************************************************************************
  (mat::BlockDense, mat::Dynamic, BCv, NBRv, NBCv),
  (mat::BlockSparse, mat::Dynamic, BCv, NBRv, NBCv),
  (mat::BlockCoeffSparse, mat::Dynamic, BCv, NBRv, NBCv),
  (mat::BlockDiagonal, mat::Dynamic, BCv, NBRv, NBCv),
  //----------------------------------------
  (mat::BlockDense, mat::Dynamic, BCv, NBRv, mat::Dynamic),
  (mat::BlockSparse, mat::Dynamic, BCv, NBRv, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::Dynamic, BCv, NBRv, mat::Dynamic),
  (mat::BlockDiagonal, mat::Dynamic, BCv, NBRv, mat::Dynamic),
  //----------------------------------------
  (mat::BlockDense, mat::Dynamic, BCv, mat::Dynamic, NBCv),
  (mat::BlockSparse, mat::Dynamic, BCv, mat::Dynamic, NBCv),
  (mat::BlockCoeffSparse, mat::Dynamic, BCv, mat::Dynamic, NBCv),
  (mat::BlockDiagonal, mat::Dynamic, BCv, mat::Dynamic, NBCv),
  //----------------------------------------
  (mat::BlockDense, mat::Dynamic, BCv, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::Dynamic, BCv, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::Dynamic, BCv, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::Dynamic, BCv, mat::Dynamic, mat::Dynamic),
  //***************************************************************************
  (mat::BlockDense, mat::Dynamic, mat::Dynamic, NBRv, NBCv),
  (mat::BlockSparse, mat::Dynamic, mat::Dynamic, NBRv, NBCv),
  (mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, NBRv, NBCv),
  (mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, NBRv, NBCv),
  //----------------------------------------
  (mat::BlockDense, mat::Dynamic, mat::Dynamic, NBRv, mat::Dynamic),
  (mat::BlockSparse, mat::Dynamic, mat::Dynamic, NBRv, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, NBRv, mat::Dynamic),
  (mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, NBRv, mat::Dynamic),
  //----------------------------------------
  (mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, NBCv),
  (mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, NBCv),
  (mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, NBCv),
  (mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, NBCv),
  //----------------------------------------
  (mat::BlockDense, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockCoeffSparse, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic),
  (mat::BlockDiagonal, mat::Dynamic, mat::Dynamic, mat::Dynamic, mat::Dynamic)
  )
{
  templateTestFunc< matType, BR, BC, NBR, NBC>();
}
