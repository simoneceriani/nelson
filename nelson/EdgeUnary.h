#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSingleSectionBase.h"

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

  template<class Section>
  class EdgeUnarySingleSection : public EdgeUnaryBase, public EdgeSingleSectionBase<Section> {

    template<class Derived, class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;

    class HessianUpdater : public EdgeHessianUpdater {
      EdgeUnarySingleSection* _e;
    public:
      HessianUpdater(EdgeUnarySingleSection* e) : _e(e) {

      }

      void updateH() override {
        _e->updateH();
      }
    };

  public:
    EdgeUnarySingleSection();
    virtual ~EdgeUnarySingleSection();

    // virtual void update(bool updateHessians) = 0; // defined in base

    virtual void updateH() = 0;

  };

  //--------------------------------------------------------------------------------------------------------

  template<class Section, class Derived>
  class EdgeUnarySingleSectionCRPT : public EdgeUnarySingleSection<Section> {

  public:

    const typename Section::ParameterType& parameter() const {
      return this->section().parameter(this->parId());
    }

    void updateH() override final {
      assert(this->HUid() >= 0);
      assert(this->parId().isVariable());

      typename Section::Hessian::MatTraits::MatrixType::BlockType bH = this->section().hessianBlockByUID(this->HUid());
      typename Section::Hessian::VecType::SegmentType bV = this->section().bVectorSegment(this->parId().id());
      static_cast<Derived*>(this)->updateHBlock(bH, bV);
    }
  };

}