#pragma once
#include "SingleSection.h"

#include "mat/VectorBlock.hpp"
#include "SingleSectionHessian.hpp"

namespace nelson {

  template<class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<ParT, matTypeV, T, B, NB>::SingleSection() {

  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<ParT, matTypeV, T, B, NB>::~SingleSection() {

  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::parametersReady() {
    assert(this->_sparsityPattern == nullptr);
    this->_sparsityPattern.reset(new mat::SparsityPattern<mat::ColMajor>(this->numParameters(), this->numParameters()));
  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::structureReady() {
    assert(this->_sparsityPattern != nullptr);
    this->_hessian.resize(this->parameterSize(), this->numParameters(), *this->_sparsityPattern);
  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::addEdge(int i/*, EdgeUnary* e*/) {
    assert(this->_sparsityPattern != nullptr);
    this->_sparsityPattern->add(i, i);
  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::addEdge(int i, int j/*, EdgeBinary* e*/) {
    assert(this->_sparsityPattern != nullptr);
    assert(i < j);
    this->_sparsityPattern->add(i, i);
    this->_sparsityPattern->add(i, j);
    this->_sparsityPattern->add(j, j);
  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::addEdge(int i, int j, int k/*, EdgeTernary* e*/) {
    assert(this->_sparsityPattern != nullptr);
    assert(i < j);
    this->_sparsityPattern->add(i, i);
    this->_sparsityPattern->add(i, j);
    this->_sparsityPattern->add(i, k);

    this->_sparsityPattern->add(j, j);
    this->_sparsityPattern->add(j, k);

    this->_sparsityPattern->add(k, k);
  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  template<int N>
  void SingleSection<ParT, matTypeV, T, B, NB>::addEdge(const std::array<int, N>& ids/*, EdgeNAry * e*/) {
    assert(this->_sparsityPattern != nullptr);
    for (int i = 0; i < ids.size(); i++) {
      for (int j = i; j < ids.size(); j++) {
        this->_sparsityPattern->add(i, j);
      }
    }

  }



}