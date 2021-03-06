#include <iostream>
#include <vector>

#include "Problems.h"

#include <chrono>
#include <fstream>

#include <unordered_set>
#include "mat/SparsityPattern.h"


int main(int argc, char* argv[]) {
  const Problem* problemPtr = &ProblemCollections::ladybug_49_7776;
  std::unique_ptr<Problem> newProblem;
  if (argc == 2) {
    newProblem.reset(new Problem());
    bool ok = newProblem->load(argv[1]);
    if (!ok) {
      std::cerr << "error reading " << argv[1] << std::endl;
      std::exit(-1);
    }
    else {
      problemPtr = newProblem.get();
    }
  }
  const Problem& problem = *problemPtr;

  mat::SparsityPattern<mat::RowMajor>::SPtr sp(new mat::SparsityPattern<mat::RowMajor>(problem.nCameras, problem.nPoints));
  for (int i = 0; i < problem.edges.size(); i++) {
    const auto& e = problem.edges[i];
    sp->add(e.first, e.second);
  }

  {
    std::vector<std::set<int>> set_orig;
    std::vector<std::vector<int>> sort_set;
    {
      auto t0 = std::chrono::steady_clock::now();
      set_orig.resize(sp->outerSize());
      for (int i = 0; i < sp->outerSize(); i++) {
        for (auto it = sp->inner(i).begin(); it != sp->inner(i).end(); it++) {
          set_orig[i].insert(*it);
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      std::cout << "set " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
    }
    {
      auto t0 = std::chrono::steady_clock::now();
      std::vector<std::unordered_set<int>> set(sp->outerSize());
      for (int i = 0; i < sp->outerSize(); i++) {
        for (auto it = sp->inner(i).begin(); it != sp->inner(i).end(); it++) {
          set[i].insert(*it);
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      sort_set.resize(sp->outerSize());
      for (int i = 0; i < sp->outerSize(); i++) {
        sort_set[i] = std::vector<int>(set[i].begin(), set[i].end());
        std::sort(sort_set[i].begin(), sort_set[i].end());
      }
      auto t2 = std::chrono::steady_clock::now();
      std::cout << "unordered_set  " << std::chrono::duration<double>(t2 - t0).count() << std::endl;
      std::cout << "  -- creation  " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
      std::cout << "  -- copy&sort " << std::chrono::duration<double>(t2 - t1).count() << std::endl;
    }

    // check
    for (int i = 0; i < sp->outerSize(); i++) {
      int c = 0;
      for (auto el : set_orig[i]) {
        assert(el == sort_set[i][c++]);
      }
    }
  }

  return 0;
}