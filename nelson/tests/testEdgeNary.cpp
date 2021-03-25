#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nelson/EdgeNary.hpp"
#include "nelson/SingleSection.hpp"
#include "mat/Global.h"

struct Edge1 : public nelson::EdgeNaryBase<1> {

  void update(bool updateHessians) override {

  }
};

struct Edge2 : public nelson::EdgeNaryBase<2> {

  void update(bool updateHessians) override {

  }
};

struct EdgeD : public nelson::EdgeNaryBase<mat::Dynamic> {

  EdgeD(int n) : nelson::EdgeNaryBase<mat::Dynamic>(n) {}

  void update(bool updateHessians) override {

  }
};

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

struct Edge1_S : public nelson::EdgeNarySectionBase<Section, 1> {

  void update(bool updateHessians) override {

  }

  void updateH(int i, int j, bool transpose) override {

  }
};

struct Edge2_S : public nelson::EdgeNarySectionBase < Section, 2> {

  void update(bool updateHessians) override {

  }

  void updateH(int i, int j, bool transpose) override {

  }
};

struct EdgeD_S : public nelson::EdgeNarySectionBase < Section, mat::Dynamic> {

  EdgeD_S(int n) : nelson::EdgeNarySectionBase < Section, mat::Dynamic>(n) {}

  void update(bool updateHessians) override {

  }

  void updateH(int i, int j, bool transpose) override {

  }
};

//struct Edge1_SC : public nelson::EdgeNarySectionBaseCRPT<Section, 1, Section::EdgeUnaryAdapter, Edge1_SC> {
struct Edge1_SC : public Section::EdgeNary<Edge1_SC, 1> {

  void update(bool updateHessians) override {

  }

  template<class Derived>
  void updateHBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {
    // with <1> will never be called
  }

  template<class Derived1, class Derived2>
  void updateHBlock(int i, Eigen::MatrixBase<Derived1>& Hij, Eigen::MatrixBase<Derived2>& b) {

  }

};

struct Edge2_SC : public nelson::EdgeNarySectionBaseCRPT<Section, 2, Section::EdgeUnaryAdapter, Edge2_SC> {

  void update(bool updateHessians) override {

  }

  template<class Derived>
  void updateHBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {
    // with <1> will never be called
  }

  template<class Derived1, class Derived2>
  void updateHBlock(int i, Eigen::MatrixBase<Derived1>& Hij, Eigen::MatrixBase<Derived2>& b) {

  }

};

struct EdgeD_SC : public nelson::EdgeNarySectionBaseCRPT<Section, mat::Dynamic, Section::EdgeUnaryAdapter, EdgeD_SC> {

  EdgeD_SC(int n) : nelson::EdgeNarySectionBaseCRPT < Section, mat::Dynamic, Section::EdgeUnaryAdapter, EdgeD_SC>(n) {}


  void update(bool updateHessians) override {

  }

  template<class Derived>
  void updateHBlock(int i, int j, Eigen::MatrixBase<Derived>& Hij, bool transpose) {
    // with <1> will never be called
  }

  template<class Derived1, class Derived2>
  void updateHBlock(int i, Eigen::MatrixBase<Derived1>& Hij, Eigen::MatrixBase<Derived2>& b) {

  }

};



TEST_CASE("testEdgeNary", "[testEdgeNary]") {
  Edge1 e1;
  Edge2 e2;
  EdgeD ed(3);

  Section s;

  Edge1_S e1_S;
  Edge2_S e2_S;
  EdgeD_S ed_S(3);

  Edge1_SC e1_SC;
  Edge2_SC e2_SC;
  EdgeD_SC ed_SC(3);

  s.addEdge({ 0 }, new Edge1_SC());
  s.addEdge({ 0,1 }, new Edge2_SC());
  s.addEdge(std::vector<nelson::NodeId>({ 0,1,2 }), new EdgeD_SC(3));

}
