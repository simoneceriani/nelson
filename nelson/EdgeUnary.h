#pragma once
#include "Global.h"

#include "EdgeInterface.h"

namespace nelson {

  class EdgeUnary : public EdgeInterface {

    int _parId;
    int _H_uid;

  public:
    EdgeUnary();
    virtual ~EdgeUnary();

    void setParId(int id);
    void setHUid(int uid);

    int parId() const {
      return _parId;
    }

    int HUid() const {
      return _H_uid;
    }

    virtual void update(bool updateHessians) = 0;

    class EdgeUIDSetter final : public EdgeUIDSetterInterface {
      EdgeUnary* _e;
    public:
      EdgeUIDSetter(EdgeUnary* e) : _e(e) {}

      void setUID(int uid) override;

    };

  };


}