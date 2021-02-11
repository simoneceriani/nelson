#include "BaseSection.h"

namespace nelson {

  void BaseSection::updateHessianBlocks(const ParallelExecSettings& settings) {

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

  void BaseSection::evaluateEdges(const ParallelExecSettings& settings, bool hessian) {

    double chi2 = 0;

    const int chunkSize = settings.chunkSize();
    const int numEval = int(this->_edgesVector.size());
    const int reqNumThread = std::min(numEval, settings.maxNumThreads());

    if (settings.isSingleThread() || reqNumThread == 1) {
      for (auto& e : _edgesVector) {
        e->update(hessian);
        chi2 += e->chi2();
      }
      this->setChi2(chi2);
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
    this->setChi2(chi2);

  }


}