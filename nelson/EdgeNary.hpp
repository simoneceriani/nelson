#pragma once
#include "EdgeNary.h"
#include "EdgeSectionBase.hpp"

namespace nelson {

  template<int N>
  EdgeNaryBase<N>::EdgeNaryBase(int n) :
    EdgeNaryContainer<N>(n),
    _H_uid(n, n)
  {
    if (N == mat::Dynamic) {
      assert(n > 0);
    }

    _H_uid.setConstant(-1);
  }

  template<int N>
  EdgeNaryBase<N>::~EdgeNaryBase() {

  }

  template<int N>
  void EdgeNaryBase<N>::setParId(int i, NodeId id) {
    this->_parId[i] = id;
  }

  template<int N>
  void EdgeNaryBase<N>::setHUid(int i, int j, int uid) {
    this->_H_uid(i, j) = uid;
  }

  //-----------------------------------------

  template<int N>
  void EdgeNaryBase<N>::EdgeUIDSetter::setUID(int uid) {
    _e->setHUid(_i, _j, uid);
  }

  //-----------------------------------------
  
  template<class Section, int N>
  EdgeNarySectionBase<Section, N>::EdgeNarySectionBase(int size) :
    EdgeNaryBase<N>(size),
    EdgeSectionBase<Section>()
  {

  }

  template<class Section, int N>
  EdgeNarySectionBase<Section, N>::~EdgeNarySectionBase() {

  }
  //-----------------------------------------

  template<class Section, int N, class SectionAdapter, class Derived>
  EdgeNarySectionBaseCRPT<Section, N, SectionAdapter, Derived>::EdgeNarySectionBaseCRPT(int size) 
    : EdgeNarySectionBase<Section, N> (size) 
  {
  }

  template<class Section, int N, class SectionAdapter, class Derived>
  EdgeNarySectionBaseCRPT<Section, N, SectionAdapter, Derived>::~EdgeNarySectionBaseCRPT() {

  }

}