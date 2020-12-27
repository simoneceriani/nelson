#include "EdgeUnary.h"
#include "EdgeUnary.hpp"

namespace nelson {
  
  EdgeUnaryBase::EdgeUnaryBase() :
    _parId(-1),
    _H_uid(-1)
  {

  }

  EdgeUnaryBase::~EdgeUnaryBase() {

  }

  void EdgeUnaryBase::setParId(int id) {
    this->_parId = id;
  }

  void EdgeUnaryBase::setHUid(int uid) {
    this->_H_uid = uid;
  }

  //-----------------------------------------

  void EdgeUnaryBase::EdgeUIDSetter::setUID(int uid) {
    _e->setHUid(uid);
  }
}