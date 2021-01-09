#pragma once
#include "SingleSection.h"

#include "mat/VectorBlock.hpp"
#include "SingleSectionHessian.hpp"

#include "EdgeUnary.hpp"

namespace nelson {

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  SingleSection<Derived, ParT, matTypeV, T, B, NB>::SingleSection() : _edgesCount(0) {

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

    // copy the edges on the vector, to allow fast 
    this->_edgesVector.resize(_edgesCount);
    int c = 0;
    for (auto& e : _edges) {
      this->_edgesVector[c++] = std::move(e);
    }
    assert(c == _edgesCount);

    // edges is no more required
    _edges.clear();
    _edgesCount = -1; // invalid


    // the _edgeSetter is no more required
    this->_edgeSetterComputer.clear();


  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(NodeId i, EdgeUnarySingleSection<Derived>* e) {
    assert(e != nullptr);

    // add to edges
    e->setParId(i);
    e->setSection(static_cast<Derived*>(this));
    this->_edges.push_front(std::unique_ptr<EdgeInterface>(e));
    this->_edgesCount++;

    // add setters
    if (i.type() == NodeType::Variable) {
      this->_edgeSetterComputer[i.id()][i.id()].list.emplace_front(SetterComputer(new EdgeUnaryBase::EdgeUIDSetter(e), new typename EdgeUnarySingleSection<Derived>::HessianUpdater(e)));
      this->_edgeSetterComputer[i.id()][i.id()].size++;
    }
  }

  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::addEdge(NodeId i, NodeId j, EdgeBinarySingleSection<Derived>* e) {

    // add to edges
    e->setPar_1_Id(i);
    e->setPar_2_Id(j);
    e->setSection(static_cast<Derived*>(this));
    this->_edges.push_front(std::unique_ptr<EdgeInterface>(e));
    this->_edgesCount++;

    // add setters
    if (i.type() == NodeType::Variable) {
      this->_edgeSetterComputer[i.id()][i.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_11_Setter(e), new typename EdgeBinarySingleSection<Derived>::HessianUpdater_11(e)));
      this->_edgeSetterComputer[i.id()][i.id()].size++;
    }

    if (i.type() == NodeType::Variable && j.type() == NodeType::Variable) {
      assert(i.id() < j.id());
      this->_edgeSetterComputer[j.id()][i.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_12_Setter(e), new typename EdgeBinarySingleSection<Derived>::HessianUpdater_12(e)));
      this->_edgeSetterComputer[j.id()][i.id()].size++;
    }

    if (j.type() == NodeType::Variable) {
      this->_edgeSetterComputer[j.id()][j.id()].list.emplace_front(SetterComputer(new EdgeBinaryBase::EdgeUID_22_Setter(e), new typename EdgeBinarySingleSection<Derived>::HessianUpdater_22(e)));
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

    double chi2 = 0;

    const auto& settings = this->settings().edgeEvalParallelSettings;
    const int chunkSize = settings.chunkSize();
    const int numEval = int(this->_edgesVector.size());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    if (settings.isSingleThread() || reqNumThread == 1) {
      for (auto& e : _edgesVector) {
        e->update(hessian);
        chi2 += e->chi2();
      }
      this->_hessian.setChi2(chi2);
    }
    else {
      // static 
      if (settings.schedule() == ParallelSchedule::schedule_static) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static) reduction(+: chi2)
          for (int i = 0; i < _edgesVector.size(); i++) {
            _edgesVector[i]->update(hessian);
            chi2 += _edgesVector[i]->chi2();
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize) reduction(+: chi2)
          for (int i = 0; i < _edgesVector.size(); i++) {
            _edgesVector[i]->update(hessian);
            chi2 += _edgesVector[i]->chi2();
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic) reduction(+: chi2)
          for (int i = 0; i < _edgesVector.size(); i++) {
            _edgesVector[i]->update(hessian);
            chi2 += _edgesVector[i]->chi2();
          }

        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize) reduction(+: chi2)
          for (int i = 0; i < _edgesVector.size(); i++) {
            _edgesVector[i]->update(hessian);
            chi2 += _edgesVector[i]->chi2();
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_guided) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided) reduction(+: chi2)
          for (int i = 0; i < _edgesVector.size(); i++) {
            _edgesVector[i]->update(hessian);
            chi2 += _edgesVector[i]->chi2();
          }

        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize) reduction(+: chi2)
          for (int i = 0; i < _edgesVector.size(); i++) {
            _edgesVector[i]->update(hessian);
            chi2 += _edgesVector[i]->chi2();
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime) reduction(+: chi2)
        for (int i = 0; i < _edgesVector.size(); i++) {
          _edgesVector[i]->update(hessian);
          chi2 += _edgesVector[i]->chi2();
        }

      }
    }
    this->_hessian.setChi2(chi2);

  }


  template<class Derived, class ParT, int matTypeV, class T, int B, int NB >
  void SingleSection<Derived, ParT, matTypeV, T, B, NB>::updateHessianBlocks() {

    const auto& settings = this->settings().hessianUpdateParallelSettings;
    const int chunkSize = settings.chunkSize();
    const int numEval = int(this->_computationUnits.size());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    if (settings.isSingleThread() || reqNumThread == 1) {
      for (int j = 0; j < this->_computationUnits.size(); j++) {
        for (auto& set : this->_computationUnits[j]) {
          set->updateH();
        }
      }
    }
    else {
      // static 
      if (settings.schedule() == ParallelSchedule::schedule_static) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static)
          for (int j = 0; j < this->_computationUnits.size(); j++) {
            for (auto& set : this->_computationUnits[j]) {
              set->updateH();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(static, chunkSize)
          for (int j = 0; j < this->_computationUnits.size(); j++) {
            for (auto& set : this->_computationUnits[j]) {
              set->updateH();
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_dynamic) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic)
          for (int j = 0; j < this->_computationUnits.size(); j++) {
            for (auto& set : this->_computationUnits[j]) {
              set->updateH();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(dynamic, chunkSize)
          for (int j = 0; j < this->_computationUnits.size(); j++) {
            for (auto& set : this->_computationUnits[j]) {
              set->updateH();
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_guided) {
        if (settings.isChunkAuto()) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided)
          for (int j = 0; j < this->_computationUnits.size(); j++) {
            for (auto& set : this->_computationUnits[j]) {
              set->updateH();
            }
          }
        }
        else {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(guided, chunkSize)
          for (int j = 0; j < this->_computationUnits.size(); j++) {
            for (auto& set : this->_computationUnits[j]) {
              set->updateH();
            }
          }

        }
      }
      else if (settings.schedule() == ParallelSchedule::schedule_runtime) {
#pragma omp parallel for num_threads(reqNumThread) default (shared) schedule(runtime)
        for (int j = 0; j < this->_computationUnits.size(); j++) {
          for (auto& set : this->_computationUnits[j]) {
            set->updateH();
          }
        }
      }
    }

  }
}