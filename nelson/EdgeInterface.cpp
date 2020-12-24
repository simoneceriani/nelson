#include "EdgeInterface.h"

namespace nelson {
  template<class T>
  EdgeInterface<T>::EdgeInterface() {

  }

  template<class T>
  EdgeInterface<T>::~EdgeInterface() {

  }

  // explicit instantiation
  template class EdgeInterface<float>;
  template class EdgeInterface<double>;

}