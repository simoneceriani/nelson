#pragma once

#include <cassert>
#include <omp.h>

namespace nelson {

  enum class ParallelSchedule {
    schedule_static,
    schedule_dynamic,
    schedule_guided,
    schedule_runtime
  };

  class ParallelExecSettings {
    int _maxNumThreads;
    int _chunkSize; // 0 to automatic
    ParallelSchedule _schedule;

  public:

    ParallelExecSettings() : 
      _maxNumThreads(1),
      _chunkSize(0), // auto
      _schedule(ParallelSchedule::schedule_static)
    {

    }

    int maxNumThreads() const { 
      return _maxNumThreads; 
    }
    
    bool isSingleThread() const {
      return _maxNumThreads == 1;
    }
    bool isMultiThread() const {
      return _maxNumThreads != 1;
    }

    int chunkSize() const {
      return _chunkSize;
    }
    bool isChunkAuto() const {
      return _chunkSize == 0;
    }

    ParallelSchedule schedule() const {
      return _schedule;
    }

    void setNumThreads(int n) {
      assert(n >= 1);
      _maxNumThreads = n;
    }
    void setSingleThread() {
      _maxNumThreads = 1;
    }
    void setNumThreadsMax() {
      _maxNumThreads = maxSupportedThreads();
    }

    static int maxSupportedThreads() {
      return omp_get_max_threads();
    }
    
    void setChunkSize(int c) {
      assert(c >= 0);
      _chunkSize = c;
    }

    void setChunkAuto() {
      _chunkSize = 0;
    }

  };

}