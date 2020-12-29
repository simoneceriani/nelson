#include "nelson/Global.h"
#include "nelson/SingleSectionHessian.hpp"

#include <omp.h>
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>

double parForInner(int id, const std::vector<double>& v1, const std::vector<double>& v2);
double parForOuter(const std::vector < std::vector<double>>& v1, const std::vector < std::vector<double>>& v2);

double parForOuter(const std::vector < std::vector<double>>& v1, const std::vector < std::vector<double>>& v2) {
  double tmp(0);
#pragma omp parallel for reduction(+:tmp) schedule(static)
  for (int i = 0; i < v1.size(); i++) {
    tmp += parForInner(i, v1[i], v2[i]);
  }
  return tmp;

}

double parForInner(int id, const std::vector<double>& v1, const std::vector<double>& v2) {
  auto entryId = std::this_thread::get_id();
  std::atomic<bool> parallel(false);
  double tmp(0);
#pragma omp parallel for reduction(+:tmp) 
  for (int i = 0; i < v1.size(); i++) {
    tmp += v1[i] + v2[i];
    if (std::this_thread::get_id() != entryId) {
      parallel = true;
    }
  }

  if (parallel) {
    std::cout << "[" << id << "] executed in PARALLEL" << std::endl;
  }
  else {
    std::cout << "[" << id << "] executed SEQUENTIAL" << std::endl;
  }
  return tmp;
}

int main(int argc, char* argv[]) {

  int outerSize = 100;
  int innerSize = 100;
  std::vector< std::vector <double>> v1(outerSize), v2(outerSize);
  for (int o = 0; o < outerSize; o++) {
    v1[o].resize(innerSize);
    v2[o].resize(innerSize);
    for (int i = 0; i < innerSize; i++) {
      v1[o][i] = rand();
      v2[o][i] = rand();
    }
  }

  std::cout << "--------- NESTED ---------" << std::endl;
  parForOuter(v1, v2); // inner will be executed sequentially, good

  std::cout << "--------- NOT NESTED ---------" << std::endl;
  parForInner(0, v1[0], v2[0]); // in this case will be executed in parallel, no external parallel section

  return 0;
}