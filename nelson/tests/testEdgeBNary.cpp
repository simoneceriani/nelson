#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/EdgeBNary.hpp"
#include "nelson/SingleSection.hpp"
#include "mat/Global.h"

TEST_CASE("testEdgeBNaryContainer", "[testEdgeNary]") {
  {
    nelson::EdgeBNaryContainer<1, 1> e(1, 1);
    REQUIRE(e.numParams1() == 1);
    REQUIRE(e.par1().numParams() == 1);
    REQUIRE(e.par1().parId().size() == 1);

    REQUIRE(e.numParams2() == 1);
    REQUIRE(e.par2().numParams() == 1);
    REQUIRE(e.par2().parId().size() == 1);
  }
  {
    nelson::EdgeBNaryContainer<1, 2> e(1, 2);
    REQUIRE(e.numParams1() == 1);
    REQUIRE(e.par1().numParams() == 1);
    REQUIRE(e.par1().parId().size() == 1);

    REQUIRE(e.numParams2() == 2);
    REQUIRE(e.par2().numParams() == 2);
    REQUIRE(e.par2().parId().size() == 2);
  }
  {
    nelson::EdgeBNaryContainer<1, Eigen::Dynamic> e(1, 5);
    REQUIRE(e.numParams1() == 1);
    REQUIRE(e.par1().numParams() == 1);
    REQUIRE(e.par1().parId().size() == 1);

    REQUIRE(e.numParams2() == 5);
    REQUIRE(e.par2().numParams() == 5);
    REQUIRE(e.par2().parId().size() == 5);
  }
  {
    nelson::EdgeBNaryContainer<Eigen::Dynamic, Eigen::Dynamic> e(7, 5);
    REQUIRE(e.numParams1() == 7);
    REQUIRE(e.par1().numParams() == 7);
    REQUIRE(e.par1().parId().size() == 7);

    REQUIRE(e.numParams2() == 5);
    REQUIRE(e.par2().numParams() == 5);
    REQUIRE(e.par2().parId().size() == 5);
  }
}

TEST_CASE("testEdgeBNaryBase", "[testEdgeNary]") {
  {
    using Base = nelson::EdgeBNaryBase<1, 1>;
    class E : public Base {
    public:
      E(int n1, int n2) : Base(1, 1) {}
      void update(bool updateHessians) override {

      }
    } e(1, 1);
  }
}

//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------

struct Element {
  static constexpr int N = 2;
};

class Section : public nelson::SingleSection<Section, Element, mat::BlockDense, double, Element::N, 3> {
  std::vector<Element> _elements;
public:

  Section() {
    this->parametersReady();
  }

  virtual const Element& parameter(nelson::NodeId i) const {
    return _elements[i.id()];
  }
  virtual Element& parameter(nelson::NodeId i) {
    return _elements[i.id()];
  }

};

template<int N1, int N2>
struct Edge1_S : public nelson::EdgeBNarySectionBaseCRPT<Section, N1, N2, Section::EdgeBinaryAdapter, Edge1_S<N1,N2>> {

  void update(bool hessians) {

  }

  template<class Derived1, class Derived2>
  void updateHUBlock(int i, Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {

  }
  template<class Derived>
  void updateHUBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {

  }
  template<class Derived1, class Derived2>
  void updateHVBlock(int i, Eigen::MatrixBase<Derived1>& H, Eigen::MatrixBase<Derived2>& b) {

  }
  template<class Derived>
  void updateHVBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {

  }
  template<class Derived>
  void updateHWBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {

  }
};

TEST_CASE("testEdgeNary", "[testEdgeNary]") {

  Edge1_S<1,1> e1_S;
  Section s;

  s.addEdge({0}, {1}, new Edge1_S<1,1>());
  s.addEdge(std::array<nelson::NodeId, 1>({0}), {1}, new Edge1_S<1,mat::Dynamic>());
  s.addEdge({0}, std::array<nelson::NodeId, 1>({1}), new Edge1_S<mat::Dynamic,1>());
  s.addEdge({0}, {1}, new Edge1_S<mat::Dynamic,mat::Dynamic>());

}