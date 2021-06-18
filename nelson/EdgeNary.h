#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSectionBase.h"

#include <array>
#include <vector>

#include "mat/Global.h"

#include <Eigen/Core>

namespace nelson {

  template<int N>
  class EdgeNaryContainer {
    std::array<NodeId, N> _parId;
  public:
    EdgeNaryContainer(int n) { std::fill(_parId.begin(), _parId.end(), -1); }

    std::array<NodeId, N>& parId() {
      return _parId;
    }

    const std::array<NodeId, N>& parId() const {
      return _parId;
    }

    constexpr int numParams() const { return N; }
  };
  template<>
  class EdgeNaryContainer<mat::Dynamic> {
    std::vector<NodeId> _parId;
  public:
    EdgeNaryContainer(int n) : _parId(n,-1) {}

    std::vector<NodeId>& parId() {
      return _parId;
    }

    const std::vector<NodeId>& parId() const {
      return _parId;
    }

    int numParams() const { return _parId.size(); }
  };

  template<int N>
  class EdgeNaryBase : public EdgeInterface  {
    EdgeNaryContainer<N> _parIds;
    Eigen::Matrix<int, N, N> _H_uid; //note, only triag up used.... can we do better storing a linear vector only?

  protected:
    void setParId(int i, NodeId id);
    void setHUid(int i, int j, int uid);

    class EdgeUIDSetter final : public EdgeUIDSetterInterface {
      EdgeNaryBase* _e;
      int _i, _j;
    public:
      EdgeUIDSetter(EdgeNaryBase* e, int i, int j) : _e(e), _i(i), _j(j) {}

      void setUID(int uid) override;

    };

  public:

    // TODO: better if 2 constructor with a enable_if like mechanism....
    EdgeNaryBase(int size = N);
    virtual ~EdgeNaryBase();

    NodeId parId(int i) const {
      return this->_parIds.parId()[i];
    }

    int HUid(int i, int j) const {
      assert(j >= i);
      return _H_uid(i,j);
    }

    int numParams() const {
      return _parIds.numParams();
    }

  };

  
  //--------------------------------------------------------------------------------------------------------

  template<class Section, int N>
  class EdgeNarySectionBase : public EdgeNaryBase<N>, public EdgeSectionBase<Section> {

    template<class Derived, class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;
    template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv> friend class DoubleSection;

    class HessianUpdater : public EdgeHessianUpdater {
      EdgeNarySectionBase* _e;
      int _i, _j;
    public:
      HessianUpdater(EdgeNarySectionBase* e, int i, int j) : _e(e), _i(i), _j(j) {

      }

      void updateH(bool transpose) override {
        _e->updateH(_i, _j,transpose);
      }
    };

  public:    

    EdgeNarySectionBase(int size = N);
    virtual ~EdgeNarySectionBase();

    // virtual void update(bool updateHessians) = 0; // defined in base

    virtual void updateH(int i, int j, bool transpose) = 0;

  };
  
  //--------------------------------------------------------------------------------------------------------

  template<class Section, int N, class SectionAdapter, class Derived>
  class EdgeNarySectionBaseCRPT : public EdgeNarySectionBase<Section, N> {

  public:
    EdgeNarySectionBaseCRPT(int size = N);
    virtual ~EdgeNarySectionBaseCRPT();


    inline const typename SectionAdapter::ParameterType& parameter(int i) const {
      return SectionAdapter::parameter(this->section(), this->parId(i));
    }

    void updateH(int i, int j, bool transpose) override final {
      assert(this->HUid(i,j) >= 0);
      assert(this->parId(i).isVariable());
      assert(this->parId(j).isVariable());

      typename SectionAdapter::HBlockType bH = SectionAdapter::HBlock(this->section(), this->HUid(i,j));
      if (i == j) {
        assert(transpose == false);
        typename SectionAdapter::BSegmentType bV = SectionAdapter::bSegment(this->section(), this->parId(i).id());
        static_cast<Derived*>(this)->updateHBlock(i, bH, bV);
      }
      else {
        static_cast<Derived*>(this)->updateHBlock(i, j, bH, transpose);
      }

    }
  };  
}