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

constexpr int N = 10;

struct Variable {

  Eigen::VectorXd var;

  Variable() {

  }

  void resize(int n) {
    var = Eigen::VectorXd::LinSpaced(n, 0, n - 1);
  }

  int size() const {
    return var.size();
  }

};

class Section : public nelson::SingleSection<Section, Variable, mat::BlockCoeffSparse, double, mat::Variable, mat::Dynamic> {

  std::vector<Variable> _variables;
  std::vector <int> _sizes;

public:

  Section() {
    _variables.resize(N);
    for (int i = 0; i < _variables.size(); i++) {
      _variables[i].resize(_variables.size() - i );
    }
    _sizes.resize(_variables.size());
    for (int i = 0; i < _variables.size(); i++) {
      _sizes[i] = _variables[i].size();
    }
  }

  const Variable& parameter(nelson::NodeId id) const override {
    return _variables[id.id()];
  }
  Variable& parameter(nelson::NodeId id) override {
    return _variables[id.id()];
  }

  const std::vector<int>& parameterSize() const {
    return _sizes;
  }

  virtual ~Section() {

  }
};

class Edge : public Section::EdgeBinary<Edge> {

  Eigen::MatrixXd J1, J2;
  Eigen::VectorXd err;

public:
  void update(bool hessians) override {
    const auto& v1 = this->parameter_1();
    const auto& v2 = this->parameter_2();

    int minN = std::min(v1.size(), v2.size());

    err = v1.var.head(minN) - v2.var.head(minN);
    err.array() += 1;

    this->setChi2(err.squaredNorm());

    J1.resize(err.size(), v1.size());
    J1.setIdentity();
    J2.resize(err.size(), v2.size());
    J2.setIdentity();
  }

  template<class Derived1, class Derived2>
  void updateH11Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {
    H.noalias() += J1.transpose() * J1;
    b.noalias() += J1.transpose() * err;
  }
  template<class Derived>
  void updateH12Block(Eigen::MatrixBase<Derived>& H, bool transpose) {
    if (!transpose) {
      H.noalias() += J1.transpose() * J2;
    }
    else {
      H.noalias() += J2.transpose() * J1;
    }
  }
  template<class Derived1, class Derived2>
  void updateH22Block(Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {
    H.noalias() += J2.transpose() * J2;
    b.noalias() += J2.transpose() * err;
  }

};

TEST_CASE("TestSingleSectionPermutation", "[]") {
  Section s;
  s.parametersReady();

  int edges = 0;
  int skip = 0;
  for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
      //if (rand() % 2 == 0) {
        s.addEdge(i, j, new Edge());
        edges++;
      //}
      //else skip++;
    }
  }

  SECTION("permute") {
    Eigen::VectorXi order = s.user2internalIndexes();
    std::random_shuffle(order.data(), order.data() + order.size());
    s.setUser2InternalIndexes(order);
  }
  SECTION("amd-permute") {
    s.permuteAMD();
  }
  SECTION("standard") {
  }

  s.structureReady();

  s.update(true);

  std::cout << s.hessian().H().mat() << std::endl;
  std::cout << s.hessian().b().mat() << std::endl;
}