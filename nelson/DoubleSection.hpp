#pragma once
#include "DoubleSection.h"

#include "mat/VectorBlock.hpp"
#include "DoubleSectionHessian.hpp"

#include "EdgeUnary.hpp"
#include "EdgeBinary.hpp"

namespace nelson {

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::DoubleSection() {

  }

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::~DoubleSection() {

  }

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::setUser2InternalIndexesU(const Eigen::Matrix<int, NBUv, 1>& v) {
    assert(_user2internalIndexesU.size() == v.size());
    this->_user2internalIndexesU = v;
  }

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::permuteAMD_SchurU() {
    // create sparsity pattern U
    mat::SparsityPattern<mat::ColMajor> spU(this->numParametersU(), this->numParametersU());
    // create sparsity pattern from edges
    for (int j = 0; j < this->_edgeSetterComputerU.size(); j++) {
      for (auto& setList : this->_edgeSetterComputerU[j]) {
        assert(_user2internalIndexesU[setList.first] == setList.first);
        assert(_user2internalIndexesU[j] == j);
        int i = setList.first;
        assert(j >= i);
        spU.add(i, j);
      }
    }
    Eigen::SparseMatrix<int> spMatU = spU.toSparseMatrix();

    // create sparsity pattern U
    mat::SparsityPattern<mat::ColMajor> spW(this->numParametersU(), this->numParametersV());
    // create sparsity pattern from edges
    for (int i = 0; i < this->_edgeSetterComputerW.size(); i++) {
      for (auto& setList : this->_edgeSetterComputerW[i]) {
        assert(_user2internalIndexesU[i] == i);
        assert(_user2internalIndexesU[setList.first] == setList.first);
        int j = setList.first;
        spW.add(i, j);
      }
    }
    Eigen::SparseMatrix<int> spMatW = spW.toSparseMatrix();

    Eigen::SparseMatrix<int> spMatUWWt = (spMatU + spMatW * spMatW.transpose()).template triangularView<Eigen::Upper>();

    Eigen::AMDOrdering<int> amd_ordering;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation;
    amd_ordering(spMatUWWt.selfadjointView<Eigen::Upper>(), permutation);
    _user2internalIndexesU = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>(permutation.transpose()).indices();
  }


  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::parametersReady() {
    auto newU = std::vector<std::map<int, ListWithCount>>(this->numParametersU());
    this->_edgeSetterComputerU.swap(newU);

    auto newV = std::vector<std::map<int, ListWithCount>>(this->numParametersV());
    this->_edgeSetterComputerV.swap(newV);

    // this is row major
    auto newW = std::vector<std::map<int, ListWithCount>>(this->numParametersU());
    this->_edgeSetterComputerW.swap(newW);

    this->_user2internalIndexesU = Eigen::Matrix<int, NBUv, 1>::LinSpaced(this->numParametersU(), 0, this->numParametersU() - 1);

  }

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::structureReady() {
    assert(this->_sparsityPatternU == nullptr);
    assert(this->_sparsityPatternV == nullptr);
    assert(this->_sparsityPatternW == nullptr);

    // create sparsity pattern from edges
    this->_sparsityPatternU.reset(new mat::SparsityPattern<mat::ColMajor>(this->numParametersU(), this->numParametersU()));
    for (int j = 0; j < this->_edgeSetterComputerU.size(); j++) {
      for (auto& setList : this->_edgeSetterComputerU[j]) {
        int newi = _user2internalIndexesU[setList.first];
        int newj = _user2internalIndexesU[j];
        if (newj > newi) {
          this->_sparsityPatternU->add(newi, newj);
        }
        else {
          this->_sparsityPatternU->add(newj, newi);
        }
        
      }
    }

    this->_sparsityPatternV.reset(new mat::SparsityPattern<mat::ColMajor>(this->numParametersV(), this->numParametersV()));
    for (int j = 0; j < this->_edgeSetterComputerV.size(); j++) {
      for (auto& setList : this->_edgeSetterComputerV[j]) {
        this->_sparsityPatternV->add(setList.first, j);
      }
    }

    this->_sparsityPatternW.reset(new mat::SparsityPattern<mat::RowMajor>(this->numParametersU(), this->numParametersV()));
    for (int i = 0; i < this->_edgeSetterComputerW.size(); i++) {
      for (auto& setList : this->_edgeSetterComputerW[i]) {
        int newi = _user2internalIndexesU[i];
        this->_sparsityPatternW->add(newi, setList.first);
      }
    }

    // prepare hessian structure
    auto parameterUSizePermuted = this->parameterUSizePermuted(_user2internalIndexesU);
    this->_hessian.resize(
      parameterUSizePermuted, this->numParametersU(),
      this->parameterVSize(), this->numParametersV(),
      this->_sparsityPatternU, this->_sparsityPatternV, this->_sparsityPatternW
    );

    //-------------------------------------------------------------------------------------------------
    // copy _edgeSetterComputer to sorted version    
    std::vector<std::map<int, ListWithCount>> _sortedEdgeSetterComputerU(this->numParametersU());
    for (int j = 0; j < this->_edgeSetterComputerU.size(); j++) {
      for (auto& setList : this->_edgeSetterComputerU[j]) {
        int newj = _user2internalIndexesU[j];
        int newi = _user2internalIndexesU[setList.first];
        if (newj >= newi) {
          _sortedEdgeSetterComputerU[newj][newi].size = setList.second.size;
          _sortedEdgeSetterComputerU[newj][newi].list.swap(setList.second.list);
          _sortedEdgeSetterComputerU[newj][newi].transpose = false;
        }
        else {
          _sortedEdgeSetterComputerU[newi][newj].size = setList.second.size;
          _sortedEdgeSetterComputerU[newi][newj].list.swap(setList.second.list);
          _sortedEdgeSetterComputerU[newi][newj].transpose = true;
        }

      }
    }
    std::vector<std::map<int, ListWithCount>> _sortedEdgeSetterComputerW(this->numParametersU());
    for (int i = 0; i < this->_edgeSetterComputerW.size(); i++) {
      int newi = _user2internalIndexesU[i];
        _sortedEdgeSetterComputerW[newi].swap(_edgeSetterComputerW[i]);
    }


    //----------------------------------------------------------------------------
    // set the UIDs to the edges
    // iterate on sparsity pattern and _edgeSetterComputer, they have to be the same!
    int buid = 0;
    for (int j = 0; j < _sortedEdgeSetterComputerU.size(); j++) {
      for (auto& setList : _sortedEdgeSetterComputerU[j]) {
        assert(_hessian.U().blockUID(setList.first, j) == buid);
        for (auto& set : setList.second.list) {
          set.setter->setUID(buid);
        }
        buid++;
      }
    }
    assert(buid == _hessian.U().nonZeroBlocks());
    
    buid = 0;
    for (int j = 0; j < this->_edgeSetterComputerV.size(); j++) {
      for (auto& setList : this->_edgeSetterComputerV[j]) {
        assert(_hessian.V().blockUID(setList.first, j) == buid);
        for (auto& set : setList.second.list) {
          set.setter->setUID(buid);
        }
        buid++;
      }
    }
    assert(buid == _hessian.V().nonZeroBlocks());

    buid = 0;
    for (int i = 0; i < _sortedEdgeSetterComputerW.size(); i++) {
      for (auto& setList : _sortedEdgeSetterComputerW[i]) {
        assert(_hessian.W().blockUID(i, setList.first) == buid);
        for (auto& set : setList.second.list) {
          set.setter->setUID(buid);
        }
        buid++;
      }
    }
    assert(buid == _hessian.W().nonZeroBlocks());

   // prepare the _computationUnits, they are the same than the H blocks
    auto newV = std::vector<UpdaterVectorWithTransposeFlag>(this->_sparsityPatternU->count() + this->_sparsityPatternV->count() + this->_sparsityPatternW->count());
    _computationUnits.swap(newV);

    int cuCount = 0;
    for (int j = 0; j < _sortedEdgeSetterComputerU.size(); j++) {
      for (auto& setList : _sortedEdgeSetterComputerU[j]) {
        _computationUnits[cuCount].updaters.resize(setList.second.size);
        _computationUnits[cuCount].transpose = setList.second.transpose;
        int c = 0;
        for (auto& set : setList.second.list) {
          //auto ptr = std::move(set.computer);
          _computationUnits[cuCount].updaters[c] = std::move(set.computer);
          c++;
        }
        cuCount++;
      }
    }
    for (int j = 0; j < this->_edgeSetterComputerV.size(); j++) {
      for (auto& setList : this->_edgeSetterComputerV[j]) {
        _computationUnits[cuCount].updaters.resize(setList.second.size);
        _computationUnits[cuCount].transpose = setList.second.transpose;
        int c = 0;
        for (auto& set : setList.second.list) {
          //auto ptr = std::move(set.computer);
          _computationUnits[cuCount].updaters[c] = std::move(set.computer);
          c++;
        }
        cuCount++;
      }
    }
    for (int i = 0; i < _sortedEdgeSetterComputerW.size(); i++) {
      for (auto& setList : _sortedEdgeSetterComputerW[i]) {
        _computationUnits[cuCount].updaters.resize(setList.second.size);
        _computationUnits[cuCount].transpose = setList.second.transpose;
        int c = 0;
        for (auto& set : setList.second.list) {
          //auto ptr = std::move(set.computer);
          _computationUnits[cuCount].updaters[c] = std::move(set.computer);
          c++;
        }
        cuCount++;
      }
    }

    assert(cuCount == this->_sparsityPatternU->count() + this->_sparsityPatternV->count() + this->_sparsityPatternW->count());
    //----------------------------------------------------------------------------

    // the _edgeSetter is no more required
    this->_edgeSetterComputerU.clear();
    this->_edgeSetterComputerV.clear();
    this->_edgeSetterComputerW.clear();

  }

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  template<class EdgeUnaryAdapter, class EdgeDerived>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::addEdge(NodeId i, EdgeUnary<EdgeUnaryAdapter, EdgeDerived>* e) {
    assert(e != nullptr);

    // add to edges
    e->setParId(i);
    e->setSection(static_cast<Derived*>(this));
    this->_edgesVector.push_back(std::unique_ptr<EdgeInterface>(e));

    // add setters
    if (i.type() == NodeType::Variable) {
      EdgeUnaryAdapter::edgeSetterComputer(*static_cast<Derived*>(this))[i.id()][i.id()].list.emplace_front(SetterComputer(new EdgeUnaryBase::EdgeUIDSetter(e), new typename EdgeUnarySectionBase<Derived>::HessianUpdater(e)));
      EdgeUnaryAdapter::edgeSetterComputer(*static_cast<Derived*>(this))[i.id()][i.id()].size++;
    }
  }

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  template<class EdgeBinaryAdapterT, class EdgeDerived>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::addEdge(NodeId i, NodeId j, EdgeBinary<EdgeBinaryAdapterT, EdgeDerived>* e) {
    assert(e != nullptr);

    // add to edges
    e->setPar_1_Id(i);
    e->setPar_2_Id(j);
    e->setSection(static_cast<Derived*>(this));
    this->_edgesVector.push_back(std::unique_ptr<EdgeInterface>(e));

    // add setters
    if (i.type() == NodeType::Variable) {
      EdgeBinaryAdapterT::edgeSetterComputer1(*static_cast<Derived*>(this))[i.id()][i.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_11_Setter(e), new typename EdgeBinarySectionBase<Derived>::HessianUpdater_11(e)));
      EdgeBinaryAdapterT::edgeSetterComputer1(*static_cast<Derived*>(this))[i.id()][i.id()].size++;
    }

    if (i.type() == NodeType::Variable && j.type() == NodeType::Variable) {
      if (!EdgeBinaryAdapterT::isEdgeAcrossSections) {
        assert(i.id() < j.id()); 
      }
      EdgeBinaryAdapterT::edgeSetterComputer12(*static_cast<Derived*>(this), i.id(), j.id()).list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_12_Setter(e), new typename EdgeBinarySectionBase<Derived>::HessianUpdater_12(e)));
      EdgeBinaryAdapterT::edgeSetterComputer12(*static_cast<Derived*>(this), i.id(), j.id()).size++;
    }

    if (j.type() == NodeType::Variable) {
      EdgeBinaryAdapterT::edgeSetterComputer2(*static_cast<Derived*>(this))[j.id()][j.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_22_Setter(e), new typename EdgeBinarySectionBase<Derived>::HessianUpdater_22(e)));
      EdgeBinaryAdapterT::edgeSetterComputer2(*static_cast<Derived*>(this))[j.id()][j.id()].size++;
    }
  }

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::update(bool hessian) {

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

  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::evaluateEdges(bool hessian) {
    BaseSection::evaluateEdges(settings().edgeEvalParallelSettings, hessian);
  }


  template<class Derived, class ParUT, class ParVT, int matTypeUv, int matTypeVv, int matTypeWv, class Tv, int BUv, int BVv, int NBUv, int NBVv>
  void DoubleSection<Derived, ParUT, ParVT, matTypeUv, matTypeVv, matTypeWv, Tv, BUv, BVv, NBUv, NBVv>::updateHessianBlocks() {
    BaseSection::updateHessianBlocks(settings().hessianUpdateParallelSettings);
  }

}