#include "EdgeUnary.h"
#include "EdgeUnary.hpp"

namespace nelson {

  EdgeUnary::EdgeUnary() :
    _parId(-1),
    _H_uid(-1)
  {

  }

  EdgeUnary::~EdgeUnary() {

  }

  void EdgeUnary::setParId(int id) {
    this->_parId = id;
  }


  void EdgeUnary::setHUid(int uid) {
    this->_H_uid = uid;
  }

  //-----------------------------------------------------------------------------

  void EdgeUnary::EdgeUIDSetter::setUID(int uid) {
    _e->setHUid(uid);
  }



}