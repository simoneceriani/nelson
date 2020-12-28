#pragma once
#include "SingleSection.h"

#include "mat/VectorBlock.hpp"
#include "SingleSectionHessian.hpp"

#include "EdgeUnary.hpp"

namespace nelson {

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<Derived, ParT, matTypeV, T, B, NB>::SingleSection() {

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<Derived, ParT, matTypeV, T, B, NB>::~SingleSection() {

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::parametersReady() {
    auto newV = std::vector<std::map<int, std::forward_list<SetterComputer>>>(this->numParameters());
    this->_edgeSetterComputer.swap(newV);
  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::structureReady() {
    assert(this->_sparsityPattern == nullptr);
    this->_sparsityPattern.reset(new mat::SparsityPattern<mat::ColMajor>(this->numParameters(), this->numParameters()));
    // create sparsity pattern from edges
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        this->_sparsityPattern->add(setList.first, j);
      }
    }

    this->_hessian.resize(this->parameterSize(), this->numParameters(), *this->_sparsityPattern);

    // iterate on sparsity pattern and _edgeSetterComputer, they have to be the same!
    int buid = 0;
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        assert(_hessian.H().blockUID(setList.first, j) == buid);
        for (auto& set : setList.second) {
          set.setter->setUID(buid);
        }
        buid++;
      }
    }
    assert(buid == _hessian.H().nonZeroBlocks());

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(int i, EdgeUnarySingleSection<Derived>* e) {
    assert(e != nullptr);

    // add to edges
    e->setParId(i);
    e->setSection(static_cast<Derived*>(this));
    this->_edges.push_front(std::unique_ptr<EdgeInterface>(e));

    // add setters
    this->_edgeSetterComputer[i][i].emplace_front(SetterComputer(new EdgeUnaryBase::EdgeUIDSetter(e),new typename EdgeUnarySingleSection<Derived>::HessianUpdater(e) ));
  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(int i, int j, EdgeBinarySingleSection<Derived>* e) {
    assert(i < j);

    // add to edges
    e->setPar_1_Id(i);
    e->setPar_2_Id(j);
    e->setSection(static_cast<Derived*>(this));
    this->_edges.push_front(std::unique_ptr<EdgeInterface>(e));

    // add setters
    this->_edgeSetterComputer[i][i].emplace_front(SetterComputer( new EdgeBinaryBase::EdgeUID_11_Setter(e),new typename EdgeBinarySingleSection<Derived>::HessianUpdater_11(e) ));

    this->_edgeSetterComputer[j][j].emplace_front(SetterComputer( new EdgeBinaryBase::EdgeUID_22_Setter(e),new typename EdgeBinarySingleSection<Derived>::HessianUpdater_12(e) ));

    this->_edgeSetterComputer[j][i].emplace_front(SetterComputer( new EdgeBinaryBase::EdgeUID_12_Setter(e),new typename EdgeBinarySingleSection<Derived>::HessianUpdater_22(e) ));
  }

  //template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  //void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(int i, int j, int k/*, EdgeTernary* e*/) {
  //  assert(i < j);
  //  assert(j < k);
  //}
  //
  //template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  //template<int N>
  //void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(const std::array<int, N>& ids/*, EdgeNAry * e*/) {
  //
  //}

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::update(bool hessian) {

    if (hessian) {
      this->_hessian.clearAll();
    }
    else {
      this->_hessian.clearChi2();
    }

    // dummy implementation so far
    for (auto& e : _edges) {
      e->update(hessian);
    }

    // test hessian updatee
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        for (auto& set : setList.second) {
          set.computer->updateH();
        }
      }
    }

  }


}