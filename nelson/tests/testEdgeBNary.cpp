#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/EdgeBNary.hpp"
#include "nelson/SingleSection.hpp"
#include "nelson/DoubleSection.hpp"
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
  std::array<Element,3> _elements;
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

class DoubleSection : public nelson::DoubleSection< DoubleSection, Element, Element, mat::BlockDense, mat::BlockDense, mat::BlockDense, double, Element::N, Element::N, 2, 3> {
  std::array<Element,2> _elements1;
  std::array<Element,3> _elements2;
public:

  DoubleSection() {
    this->parametersReady();
  }

  virtual const Element& parameterU(nelson::NodeId i) const override {
    return _elements1[i.id()];
  }
  virtual Element& parameterU(nelson::NodeId i) override {
    return _elements2[i.id()];

  }

  virtual const Element& parameterV(nelson::NodeId i) const override {
    return _elements2[i.id()];
  }
  virtual Element& parameterV(nelson::NodeId i) override {
    return _elements2[i.id()];
  }
};

template<class SectionBase, class Adapter, int N1, int N2>
struct Edge1_S : public nelson::EdgeBNarySectionBaseCRPT<SectionBase, N1, N2, Adapter, Edge1_S<SectionBase, Adapter, N1,N2>> {

  Edge1_S(int n1 = N1, int n2 = N2) 
    : nelson::EdgeBNarySectionBaseCRPT<SectionBase, N1, N2, Adapter, Edge1_S<SectionBase, Adapter, N1, N2>>(n1,n2)
  {

  }

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

TEST_CASE("testEdgeNary-singleSection", "[testEdgeNary]") {

  Edge1_S<Section, Section::EdgeBinaryAdapter, 1, 1> e1_S(1,1);
  Section s;

  s.addEdge({0}, {1}, new Edge1_S<Section, Section::EdgeBinaryAdapter, 1,1>());
  //new Edge1_S<1, mat::Dynamic>(1,1);
  s.addEdge({0}, {1}, new Edge1_S<Section, Section::EdgeBinaryAdapter, 1, mat::Dynamic>(1,1));
  s.addEdge({0}, {1}, new Edge1_S<Section, Section::EdgeBinaryAdapter, mat::Dynamic,1>(1,1));
  s.addEdge({0}, {1}, new Edge1_S<Section, Section::EdgeBinaryAdapter, mat::Dynamic,mat::Dynamic>(1,1));

} 

TEST_CASE("testEdgeNary-doubleSection", "[testEdgeNary]") {

  Edge1_S<DoubleSection, DoubleSection::EdgeBinaryAdapterUV, 1, 1> e1_S(1,1);
  DoubleSection s;

  s.addEdge({0}, {1}, new Edge1_S<DoubleSection, DoubleSection::EdgeBinaryAdapterUV, 1,1>());
  s.addEdge({0}, {1}, new Edge1_S<DoubleSection, DoubleSection::EdgeBinaryAdapterUV, 1, mat::Dynamic>(1,1));
  s.addEdge({0}, {1}, new Edge1_S<DoubleSection, DoubleSection::EdgeBinaryAdapterUV, mat::Dynamic,1>(1,1));
  s.addEdge({0}, {1}, new Edge1_S<DoubleSection, DoubleSection::EdgeBinaryAdapterUV, mat::Dynamic,mat::Dynamic>(1,1));


} 