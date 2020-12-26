#pragma once

namespace nelson {

  class EdgeInterface {

  public:
    EdgeInterface();
    virtual ~EdgeInterface();
    virtual void update(bool updateHessians) = 0;

  };

  class EdgeUIDSetterInterface {
  public:
    virtual void setUID(int uid) = 0;
  };

}