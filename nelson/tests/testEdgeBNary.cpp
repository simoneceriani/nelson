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