#include "EdgeBinary.h"
#include "EdgeBinary.hpp"

namespace nelson {

  EdgeBinary::EdgeBinary() :
    _par_1_Id(-1),
    _par_2_Id(-1),
    _H_11_uid(-1),
    _H_12_uid(-1),
    _H_22_uid(-1)
  {

  }

  EdgeBinary::~EdgeBinary() {

  }

  void EdgeBinary::setPar_1_Id(int id) {
    this->_par_1_Id = id;
  }
  void EdgeBinary::setPar_2_Id(int id) {
    this->_par_2_Id = id;
  }


  void EdgeBinary::setH_11_Uid(int uid) {
    this->_H_11_uid = uid;
  }
  void EdgeBinary::setH_12_Uid(int uid) {
    this->_H_12_uid = uid;
  }
  void EdgeBinary::setH_22_Uid(int uid) {
    this->_H_22_uid = uid;
  }

  //-----------------------------------------------------------------------------

  void EdgeBinary::EdgeUID_11_Setter::setUID(int uid) {
    _e->setH_11_Uid(uid);
  }
  void EdgeBinary::EdgeUID_12_Setter::setUID(int uid) {
    _e->setH_12_Uid(uid);
  }
  void EdgeBinary::EdgeUID_22_Setter::setUID(int uid) {
    _e->setH_22_Uid(uid);
  }



}