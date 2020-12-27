#pragma once
#include "SingleSection.h"

#include "mat/VectorBlock.hpp"
#include "SingleSectionHessian.hpp"

#include "EdgeUnary.hpp"

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
    this->_edgeSetter.swap(std::vector<std::map<int, std::forward_list<std::unique_ptr<EdgeUIDSetterInterface>>>>(this->numParameters()));
  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::structureReady() {
    assert(this->_sparsityPattern != nullptr);
    this->_hessian.resize(this->parameterSize(), this->numParameters(), *this->_sparsityPattern);

    // iterate on sparsity pattern and _edgeSetter, they have to be the same!
    int buid = 0;
    for (int j = 0; j < this->_edgeSetter.size(); j++) {
      for (auto& setList : this->_edgeSetter[j]) {
        assert(_hessian.H().blockUID(setList.first, j) == buid);
        for (auto& set : setList.second) {
          set->setUID(buid);
        }
        buid++;
      }
    }
    assert(buid == _hessian.H().nonZeroBlocks());

  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::addEdge(int i, EdgeUnary* e) {
    assert(e != nullptr);
    assert(this->_sparsityPattern != nullptr);

    // add to sparsity pattern
    this->_sparsityPattern->add(i, i);

    // add to edges
    e->setParId(i);
    this->_edges.push_front(std::unique_ptr<EdgeInterface>(e));

    // add setters
    this->_edgeSetter[i][i].emplace_front(new EdgeUnary::EdgeUIDSetter(e));
  }

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::addEdge(int i, int j, EdgeBinary* e) {
    assert(this->_sparsityPattern != nullptr);
    assert(i < j);
    this->_sparsityPattern->add(i, i);
    this->_sparsityPattern->add(i, j);
    this->_sparsityPattern->add(j, j);

    // add to edges
    e->setPar_1_Id(i);
    e->setPar_2_Id(j);
    this->_edges.push_front(std::unique_ptr<EdgeInterface>(e));

    // add setters
    this->_edgeSetter[i][i].emplace_front(new EdgeBinary::EdgeUID_11_Setter(e));
    this->_edgeSetter[j][j].emplace_front(new EdgeBinary::EdgeUID_22_Setter(e));
    this->_edgeSetter[j][i].emplace_front(new EdgeBinary::EdgeUID_12_Setter(e));
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

  template<class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<ParT, matTypeV, T, B, NB>::update() {
    // dummy implementation so far
    for (auto& e : _edges) {
      e->update(true);
    }
  }


}