#pragma once
#include "Global.h"

#include "EdgeInterface.h"
#include "EdgeSingleSectionBase.h"

namespace nelson {

  class EdgeUnaryBase : public EdgeInterface {

    int _parId;
    int _H_uid;

  protected:
    void setParId(int id);
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

    int parId() const {
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

    void update(bool updateHessian) override {
      static_cast<Derived*>(this)->update(this->section().parameter(this->parId()), updateHessian);
    }

    void updateH() override final {
      static_cast<Derived*>(this)->updateHBlock(this->section().hessianBlockByUID(this->HUid()));
    }
  };

}