#pragma once
#include "Global.h"

#include "EdgeInterface.h"

namespace nelson {

  class EdgeUnary : public EdgeInterface {

    template<class ParT, int matTypeV, class T, int B, int NB> friend class SingleSection;

    int _parId;
    int _H_uid;

    void setParId(int id);
    void setHUid(int uid);

    class EdgeUIDSetter final : public EdgeUIDSetterInterface {
      EdgeUnary* _e;
    public:
      EdgeUIDSetter(EdgeUnary* e) : _e(e) {}

      void setUID(int uid) override;

    };

  public:
    EdgeUnary();
    virtual ~EdgeUnary();


    int parId() const {
      return _parId;
    }

    int HUid() const {
      return _H_uid;
    }

    virtual void update(bool updateHessians) = 0;

  };


}