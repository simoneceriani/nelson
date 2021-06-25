#pragma once
#include "EdgeBNary.h"
#include "EdgeSectionBase.hpp"

#include "EdgeNary.hpp"

namespace nelson {

  //-----------------------------------------

  template<int N1, int N2>
  EdgeBNaryBase<N1, N2>::EdgeBNaryBase(int n1, int n2) :
    _parIds(n1, n2),
    _HU_uid(n1, n1),
    _HW_uid(n1, n2),
    _HV_uid(n2, n2)
  {

  }
  template<int N1, int N2>
  EdgeBNaryBase<N1, N2>::~EdgeBNaryBase() {

  }


  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::setPar1Id(int i, NodeId id) {
    this->_parIds.par1().parId()[i] = id;
  }
  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::setPar2Id(int i, NodeId id) {
    this->_parIds.par2().parId()[i] = id;
  }

  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::setH_U_Uid(int i, int j, int uid) {
    this->_HU_uid(i, j) = uid;
  }
  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::setH_W_Uid(int i, int j, int uid) {
    this->_HW_uid(i, j) = uid;
  }
  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::setH_V_Uid(int i, int j, int uid) {
    this->_HV_uid(i, j) = uid;
  }

  //-----------------------------------------

  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::EdgeUID_U_Setter::setUID(int uid) {
    _e->setH_U_Uid(_i, _j, uid);
  }
  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::EdgeUID_W_Setter::setUID(int uid) {
    _e->setH_W_Uid(_i, _j, uid);
  }
  template<int N1, int N2>
  void EdgeBNaryBase<N1, N2>::EdgeUID_V_Setter::setUID(int uid) {
    _e->setH_V_Uid(_i, _j, uid);
  }

  //-----------------------------------------

  template<class Section, int N1, int N2>
  EdgeBNarySectionBase<Section, N1, N2>::EdgeBNarySectionBase(int size1, int size2) :
    EdgeBNaryBase<N1,N2>(size1,size2),
    EdgeSectionBase<Section>()
  {

  }

  template<class Section, int N1, int N2>
  EdgeBNarySectionBase<Section, N1, N2>::~EdgeBNarySectionBase() {

  }

  //-----------------------------------------

  template<class Section, int N1, int N2, class SectionAdapter, class Derived>
  EdgeBNarySectionBaseCRPT<Section, N1, N2, SectionAdapter, Derived>::EdgeBNarySectionBaseCRPT(int size1, int size2)
    : EdgeBNarySectionBase<Section, N1, N2>(size1, size2)
  {
  }

  template<class Section, int N1, int N2, class SectionAdapter, class Derived>
  EdgeBNarySectionBaseCRPT<Section, N1, N2, SectionAdapter, Derived>::~EdgeBNarySectionBaseCRPT() {

  }

}