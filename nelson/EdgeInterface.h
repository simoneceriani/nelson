#pragma once

namespace nelson {

  template<class T>
  class EdgeInterface {

  public:
    EdgeInterface();
    virtual ~EdgeInterface();
    virtual T chi2() const = 0;
    virtual void update(bool updateHessians) = 0;

  };

}