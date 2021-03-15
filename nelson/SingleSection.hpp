#pragma once
#include "SingleSection.h"

#include "mat/VectorBlock.hpp"
#include "SingleSectionHessian.hpp"

#include "EdgeUnary.hpp"
#include "EdgeBinary.hpp"

namespace nelson {

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<Derived, ParT, matTypeV, T, B, NB>::SingleSection()  {

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<Derived, ParT, matTypeV, T, B, NB>::~SingleSection() {

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::parametersReady() {
    auto newV = std::vector<std::map<int, ListWithCount>>(this->numParameters());
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

    this->_hessian.resize(this->parameterSize(), this->numParameters(), this->_sparsityPattern);

    // set the UIDs to the edges
    // iterate on sparsity pattern and _edgeSetterComputer, they have to be the same!
    int buid = 0;
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        assert(_hessian.H().blockUID(setList.first, j) == buid);
        for (auto& set : setList.second.list) {
          set.setter->setUID(buid);
        }
        buid++;
      }
    }
    assert(buid == _hessian.H().nonZeroBlocks());

    // prepare the _computationUnits, they are the same than the H blocks
    auto newV = std::vector<std::vector<std::unique_ptr<EdgeHessianUpdater>>>(this->_sparsityPattern->count());
    _computationUnits.swap(newV);

    int cuCount = 0;
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        _computationUnits[cuCount].resize(setList.second.size);
        int c = 0;
        for (auto& set : setList.second.list) {
          //auto ptr = std::move(set.computer);
          _computationUnits[cuCount][c] = std::move(set.computer);
          c++;
        }
        cuCount++;
      }
    }

    // the _edgeSetter is no more required
    this->_edgeSetterComputer.clear();


  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  template<class EdgeDerived>
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(NodeId i, EdgeUnary<EdgeDerived> * e) {
    assert(e != nullptr);

    // add to edges
    e->setParId(i);
    e->setSection(static_cast<Derived*>(this));
    this->_edgesVector.push_back(std::unique_ptr<EdgeInterface>(e));

    // add setters
    if (i.type() == NodeType::Variable) {
      this->_edgeSetterComputer[i.id()][i.id()].list.emplace_front(SetterComputer(new EdgeUnaryBase::EdgeUIDSetter(e), new typename EdgeUnarySectionBase<Derived>::HessianUpdater(e)));
      this->_edgeSetterComputer[i.id()][i.id()].size++;
    }
  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  template<class EdgeDerived>
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(NodeId i, NodeId j, EdgeBinary<EdgeDerived>* e) {

    // add to edges
    e->setPar_1_Id(i);
    e->setPar_2_Id(j);
    e->setSection(static_cast<Derived*>(this));
    this->_edgesVector.push_back(std::unique_ptr<EdgeInterface>(e));

    // add setters
    if (i.type() == NodeType::Variable) {
      this->_edgeSetterComputer[i.id()][i.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_11_Setter(e), new typename EdgeBinarySectionBase<Derived>::HessianUpdater_11(e)));
      this->_edgeSetterComputer[i.id()][i.id()].size++;
    }

    if (i.type() == NodeType::Variable && j.type() == NodeType::Variable) {
      assert(i.id() < j.id());
      this->_edgeSetterComputer[j.id()][i.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_12_Setter(e), new typename EdgeBinarySectionBase<Derived>::HessianUpdater_12(e)));
      this->_edgeSetterComputer[j.id()][i.id()].size++;
    }

    if (j.type() == NodeType::Variable) {
      this->_edgeSetterComputer[j.id()][j.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_22_Setter(e), new typename EdgeBinarySectionBase<Derived>::HessianUpdater_22(e)));
      this->_edgeSetterComputer[j.id()][j.id()].size++;
    }

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

    // edge update
    this->evaluateEdges(hessian);

    // hessian update, will call the appropriate edge functions
    if (hessian) {
      // move to a function to cover all parallel cases
      this->updateHessianBlocks();
    }

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::evaluateEdges(bool hessian) {

    BaseSection::evaluateEdges(settings().edgeEvalParallelSettings, hessian);

  }


  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::updateHessianBlocks() {
    BaseSection::updateHessianBlocks(settings().hessianUpdateParallelSettings);

  }
}