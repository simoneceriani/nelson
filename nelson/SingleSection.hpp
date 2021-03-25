#pragma once
#include "SingleSection.h"

#include "mat/VectorBlock.hpp"
#include "SingleSectionHessian.hpp"

#include "EdgeUnary.hpp"
#include "EdgeBinary.hpp"

#include <Eigen/Sparse>

namespace nelson {

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<Derived, ParT, matTypeV, T, B, NB>::SingleSection() {

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<Derived, ParT, matTypeV, T, B, NB>::~SingleSection() {

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::setUser2InternalIndexes(const Eigen::Matrix<int, NB, 1>& v) {
    assert(_user2internalIndexes.size() == v.size());
    _user2internalIndexes = v;
  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::permuteAMD() {
    // create sparsity pattern
    mat::SparsityPattern<mat::ColMajor> sp(this->numParameters(), this->numParameters());
    // create sparsity pattern from edges
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        assert(_user2internalIndexes[setList.first] == setList.first);
        assert(_user2internalIndexes[j] == j);
        int i = setList.first;
        assert(j >= i);
        sp.add(i, j);
      }
    }
    Eigen::SparseMatrix<int> spMat = sp.toSparseMatrix();
    Eigen::AMDOrdering<int> amd_ordering;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation;
    amd_ordering(spMat.selfadjointView<Eigen::Upper>(), permutation);
    _user2internalIndexes = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>(permutation.transpose()).indices();

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::parametersReady() {
    auto newV = std::vector<std::map<int, ListWithCount>>(this->numParameters());
    this->_edgeSetterComputer.swap(newV);

    _user2internalIndexes = Eigen::Matrix<int, NB, 1>::LinSpaced(this->numParameters(), 0, this->numParameters() - 1);
  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::structureReady() {
    assert(this->_sparsityPattern == nullptr);
    this->_sparsityPattern.reset(new mat::SparsityPattern<mat::ColMajor>(this->numParameters(), this->numParameters()));
    // create sparsity pattern from edges
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        int newi = _user2internalIndexes[setList.first];
        int newj = _user2internalIndexes[j];
        if (newj > newi) {
          this->_sparsityPattern->add(newi, newj);
        }
        else {
          this->_sparsityPattern->add(newj, newi);
        }
      }
    }

    this->_hessian.resize(this->parameterSizePermuted(_user2internalIndexes), this->numParameters(), this->_sparsityPattern);

    // copy _edgeSetterComputer to sorted version    
    std::vector<std::map<int, ListWithCount>> _sortedEdgeSetterComputer(this->numParameters());
    for (int j = 0; j < this->_edgeSetterComputer.size(); j++) {
      for (auto& setList : this->_edgeSetterComputer[j]) {
        int newj = _user2internalIndexes[j];
        int newi = _user2internalIndexes[setList.first];
        if (newj >= newi) {
          _sortedEdgeSetterComputer[newj][newi].size = setList.second.size;
          _sortedEdgeSetterComputer[newj][newi].list.swap(setList.second.list);
          _sortedEdgeSetterComputer[newj][newi].transpose = false;
        }
        else {
          _sortedEdgeSetterComputer[newi][newj].size = setList.second.size;
          _sortedEdgeSetterComputer[newi][newj].list.swap(setList.second.list);
          _sortedEdgeSetterComputer[newi][newj].transpose = true;
        }

      }
    }

    // set the UIDs to the edges
    // iterate on sparsity pattern and _edgeSetterComputer, they have to be the same!
    int buid = 0;
    for (int j = 0; j < _sortedEdgeSetterComputer.size(); j++) {
      for (auto& setList : _sortedEdgeSetterComputer[j]) {
        assert(_hessian.H().blockUID(setList.first, j) == buid);
        for (auto& set : setList.second.list) {
          set.setter->setUID(buid);
        }
        buid++;
      }
    }
    assert(buid == _hessian.H().nonZeroBlocks());

    // prepare the _computationUnits, they are the same than the H blocks
    auto newV = std::vector < UpdaterVectorWithTransposeFlag>(this->_sparsityPattern->count());
    _computationUnits.swap(newV);

    int cuCount = 0;
    for (int j = 0; j < _sortedEdgeSetterComputer.size(); j++) {
      for (auto& setList : _sortedEdgeSetterComputer[j]) {
        _computationUnits[cuCount].updaters.resize(setList.second.size);
        _computationUnits[cuCount].transpose = setList.second.transpose;
        int c = 0;
        for (auto& set : setList.second.list) {
          _computationUnits[cuCount].updaters[c] = std::move(set.computer);
          c++;
        }
        cuCount++;
      }
    }

    // the _edgeSetter is no more required
    _sortedEdgeSetterComputer.clear();
    this->_edgeSetterComputer.clear();


  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  template<class EdgeDerived>
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(NodeId i, EdgeUnary<EdgeDerived>* e) {
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

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  template<class EdgeDerived, int N>
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(const std::array<NodeId, N>& ids, EdgeNary<EdgeDerived, N>* e) {
    assert(ids.size() == e->numParams());

    for (int i = 0; i < ids.size(); i++) {
      e->setParId(i, ids[i]);
    }
    e->setSection(static_cast<Derived*>(this));
    this->_edgesVector.push_back(std::unique_ptr<EdgeInterface>(e));

    // add setters
    for (int ii = 0; ii < ids.size(); ii++) {
      auto i = ids[ii];

      if (i.type() == NodeType::Variable) {
        this->_edgeSetterComputer[i.id()][i.id()].list.emplace_front(SetterComputer(new typename EdgeNaryBase<N>::EdgeUIDSetter(e, ii, ii), new typename EdgeNarySectionBase<Derived, N>::HessianUpdater(e, ii, ii)));
        this->_edgeSetterComputer[i.id()][i.id()].size++;
      }

      for (int jj = ii + 1; jj < ids.size(); jj++) {
        auto j = ids[jj];

        if (i.type() == NodeType::Variable && j.type() == NodeType::Variable) {
          assert(i.id() < j.id());
          this->_edgeSetterComputer[j.id()][i.id()].list.emplace_front(SetterComputer(new typename EdgeNaryBase<N>::EdgeUIDSetter(e, ii, jj), new typename EdgeNarySectionBase<Derived, N>::HessianUpdater(e, ii, jj)));
          this->_edgeSetterComputer[j.id()][i.id()].size++;
        }
      }

    }

  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  template<class EdgeDerived>
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(const std::vector<NodeId>& ids, EdgeNary<EdgeDerived, mat::Dynamic>* e) {
    assert(ids.size() == e->numParams());

    for (int i = 0; i < ids.size(); i++) {
      e->setParId(i, ids[i]);
    }
    e->setSection(static_cast<Derived*>(this));
    this->_edgesVector.push_back(std::unique_ptr<EdgeInterface>(e));

    // add setters
    for (int ii = 0; ii < ids.size(); ii++) {
      auto i = ids[ii];

      if (i.type() == NodeType::Variable) {
        this->_edgeSetterComputer[i.id()][i.id()].list.emplace_front(SetterComputer(new typename EdgeNaryBase<mat::Dynamic>::EdgeUIDSetter(e, ii, ii), new typename EdgeNarySectionBase<Derived, mat::Dynamic>::HessianUpdater(e, ii, ii)));
        this->_edgeSetterComputer[i.id()][i.id()].size++;
      }

      for (int jj = ii + 1; jj < ids.size(); jj++) {
        auto j = ids[jj];

        if (i.type() == NodeType::Variable && j.type() == NodeType::Variable) {
          assert(i.id() < j.id());
          this->_edgeSetterComputer[j.id()][i.id()].list.emplace_front(SetterComputer(new typename EdgeNaryBase<mat::Dynamic>::EdgeUIDSetter(e, ii, jj), new typename EdgeNarySectionBase<Derived, mat::Dynamic>::HessianUpdater(e, ii, jj)));
          this->_edgeSetterComputer[j.id()][i.id()].size++;
        }
      }

    }

  }

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