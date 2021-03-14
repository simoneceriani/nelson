#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/MatrixDiagInv.hpp"
#include "nelson/MatrixDenseWrapper.hpp"

#include "mat/VectorBlock.hpp"
#include "mat/MatrixBlockDescriptor.hpp"
#include "mat/DiagonalMatrixBlock.hpp"
#include "mat/SparseCoeffDiagonalMatrixBlock.hpp"

#include <iostream>


constexpr int BCv = 3;
constexpr int NBCv = 5;

#define DEBUGOUT if(false)

template<int matType, int BC, int NBC>
void templateTestFunc() {
  mat::MatrixBlockIterableTypeTraits<matType, double, mat::ColMajor, BC, BC, NBC, NBC>::MatrixType V;
  mat::SparsityPattern<mat::ColMajor>::SPtr spV(new mat::SparsityPattern<mat::ColMajor>(NBCv, NBCv));
  spV->setDiagonal();

  auto blockDescriptorV = mat::MatrixBlockDescriptor<BC, BC, NBC, NBC>(BCv, NBCv, BCv, NBCv);
  V.resize(blockDescriptorV, spV);
  V.setZero();
  for (int bi = 0; bi < V.nonZeroBlocks(); bi++) {
    V.blockByUID(bi).setRandom();
  }

  mat::VectorBlock<double, BC, NBC> b, bv;
  b.resize(blockDescriptorV.rowDescriptionCSPtr());
  b.mat().setRandom();
  bv.resize(blockDescriptorV.rowDescriptionCSPtr());

  nelson::MatrixDiagInv<matType, double, BC, NBC, mat::BlockDiagonal> mvinv;

  SECTION("SINGLE THREAD") {
    mvinv.settings().blockInversion.setSingleThread();
    mvinv.settings().rightVectorMult.setSingleThread();
  }
  SECTION("MULTI THREAD") {
    mvinv.settings().blockInversion.setNumThreadsMax();
    mvinv.settings().rightVectorMult.setNumThreadsMax();
  }

  mvinv.init(V);
  mvinv.compute(V, 0, 0);
  mvinv.rightMultVector(b, bv);

  nelson::DenseWrapper<mat::BlockDiagonal, double, mat::ColMajor, BC, BC, NBC, NBC> Vinv_Wrap;
  Vinv_Wrap.set(&mvinv.Vinv());
  DEBUGOUT std::cout << "Vinv_Wrap" << std::endl << Vinv_Wrap.mat() << std::endl << std::endl;


  Eigen::VectorXd diff = Vinv_Wrap.mat() * b.mat() - bv.mat();
  DEBUGOUT std::cout << "diff" << std::endl << diff.transpose() << std::endl << std::endl;

  bool ok = (diff.array().abs() < Eigen::NumTraits<double>::dummy_precision()).all();
  REQUIRE(ok);

}

TEMPLATE_TEST_CASE_SIG("MatrixVinvMultiplier", "[MatrixVinvMultiplier]",
  ((int matType, int BC, int NBC), matType, BC, NBC),
  (mat::BlockDiagonal, BCv, NBCv),
  (mat::BlockDiagonal, BCv, mat::Dynamic),
  (mat::BlockDiagonal, mat::Dynamic, NBCv),
  (mat::BlockDiagonal, mat::Dynamic, mat::Dynamic),
  (mat::SparseCoeffBlockDiagonal, BCv, NBCv),
  (mat::SparseCoeffBlockDiagonal, BCv, mat::Dynamic),
  (mat::SparseCoeffBlockDiagonal, mat::Dynamic, NBCv),
  (mat::SparseCoeffBlockDiagonal, mat::Dynamic, mat::Dynamic)
)
{
  templateTestFunc<matType, BC, NBC>();
}
