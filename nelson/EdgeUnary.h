#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSectionBase.h"

namespace nelson {

  class EdgeUnaryBase : public EdgeInterface {

    NodeId _parId;
    int _H_uid;

  protected:
    void setParId(NodeId id);
    void setHUid(int uid);

    class EdgeUIDSetter final : public EdgeUIDSetterInterface {
      EdgeUnaryBase* _e;
    public:
      EdgeUIDSetter(EdgeUnaryBase* e) : _e(e) {}

      void setUID(int uid) override;

    };

  public:

    EdgeUnaryBase();
    virtual ~EdgeUnaryBase();

    NodeId parId() const {
      return _parId;
    }

    int HUid() const {
      return _H_uid;
    }

  };

  //--------------------------------------------------------------------------------------------------------

  template<class Section >
  class EdgeUnarySectionBase : public EdgeUnaryBase, public EdgeSectionBase<Section> {

    template<class Derived, class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;
    template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv> friend class DoubleSection;

    class HessianUpdater : public EdgeHessianUpdater {
      EdgeUnarySectionBase* _e;
    public:
      HessianUpdater(EdgeUnarySectionBase* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH();
      }
    };

  public:    

    EdgeUnarySectionBase();
    virtual ~EdgeUnarySectionBase();

    // virtual void update(bool updateHessians) = 0; // defined in base

    virtual void updateH() = 0;

  };

  //--------------------------------------------------------------------------------------------------------

  template<class Section, class SectionAdapter, class Derived>
  class EdgeUnarySectionBaseCRPT : public EdgeUnarySectionBase<Section> {

  public:

    inline const typename SectionAdapter::ParameterType& parameter() const {
      //return this->section().parameter(this->parId());
      return SectionAdapter::parameter(this->section(), this->parId());
    }

    void updateH() override final {
      assert(this->HUid() >= 0);
      assert(this->parId().isVariable());

      //typename Section::Hessian::MatTraits::MatrixType::BlockType bH = this->section().hessianBlockByUID(this->HUid());
      //typename Section::Hessian::VecType::SegmentType bV = this->section().bVectorSegment(this->parId().id());
      //static_cast<Derived*>(this)->updateHBlock(bH, bV);

      typename SectionAdapter::HBlockType bH = SectionAdapter::HBlock(this->section(), this->HUid());
      typename SectionAdapter::BSegmentType bV = SectionAdapter::bSegment(this->section(), this->parId().id());
      static_cast<Derived*>(this)->updateHBlock(bH, bV);

    }
  };

}