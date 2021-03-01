#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/MatrixWWtMultiplier.hpp"

constexpr int BR = 6;
constexpr int NBR = 3;

constexpr int BC = 3;
constexpr int NBC = 5;

TEST_CASE("MatrixWWtMultiplier-Base", "[MatrixWWtMultiplier-Base]") {

  mat::DenseMatrixBlockIterable<double, mat::RowMajor, BR, BC, mat::Dynamic, mat::Dynamic> W;

  mat::SparsityPattern<mat::RowMajor>::SPtr sp(new mat::SparsityPattern<mat::RowMajor>(NBR, NBC));
  // 0 1 2 3 4
  // X - - X X
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

  W.resize(mat::MatrixBlockDescriptor<BR, BC, mat::Dynamic, mat::Dynamic>(BR, BC, NBR, NBC), sp);
  for (int bi = 0; bi < W.nonZeroBlocks(); bi++) {
    W.blockByUID(bi).setRandom();
  }

  nelson::MatrixWWtMultiplier<mat::BlockDense, double, BR, BC, mat::Dynamic, mat::Dynamic, mat::BlockCoeffSparse, mat::ColMajor> mwwt;
  mwwt.prepare(W);
  mwwt.multiply(W, W);

}