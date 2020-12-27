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

  public:
    EdgeUnarySingleSection();
    virtual ~EdgeUnarySingleSection();

    // virtual void update(bool updateHessians) = 0; // defined in base

  };


}