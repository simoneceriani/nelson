cmake_minimum_required(VERSION 3.17)
project(nelson LANGUAGES CXX)

#------------------------------------------------------------------------------------------
# Declare dependencies, use as find_package
# find_package_wrapper(...)
find_package_wrapper(Eigen3 REQUIRED CONFIG)
find_package_wrapper(OpenMP REQUIRED)

if(NOT nelson_BUILD_MAT)
  find_package_wrapper(mat REQUIRED CONFIG)
endif()

#------------------------------------------------------------------------------------------
# prepare headers and sources

# define the library
add_library(nelson STATIC)
# use the target_sources_wrapper, which wrap cmake native target_sources. 
# Wrapping is necessary to cover both build and installation step
# Normally headers are PUBLIC and cpp are PRIVATE, but you might have private headers too
# If the option NO_AUTO_SOURCE_GROUP is omitted the target sources and headers are 
# automatically grouped respectively in "Sources" and "Headers" folder in the IDE
# environment whereas the dependency headers (i.e. those from linked targets) are grouped
# in the "Dependency Headers" folder.
# Specify NO_AUTO_SOURCE_GROUP if you want all sources and headers grouped in the IDE
# standard folders ("Source Files" and "Header Files" in Visual Studio).
# note: you can repeat target_sources_wrapper if you want to organize your files
target_sources_wrapper(nelson
	PUBLIC 
    Global.h
    BaseSection.h
    SingleSectionHessian.h
    SingleSectionHessian.hpp
    DoubleSectionHessianMatrices.h
    DoubleSectionHessianMatrices.hpp
    DoubleSectionHessian.h
    DoubleSectionHessian.hpp
    SingleSection.h
    SingleSection.hpp
    DoubleSection.h
    DoubleSection.hpp
    EdgeInterface.h
    EdgeSectionBase.h
    EdgeSectionBase.hpp
    EdgeNary.h
    EdgeNary.hpp
    EdgeUnary.h
    EdgeUnary.hpp
    EdgeBinary.h
    EdgeBinary.hpp
    MatrixWrapperTraits.h
    MatrixDenseWrapper.h
    MatrixDenseWrapper.hpp
    MatrixSparseWrapper.h
    MatrixSparseWrapper.hpp
    MatrixDiagInv.h
    MatrixDiagInv.hpp
    MatrixWWtMultiplier.h
    MatrixWWtMultiplier.hpp
    MatrixWVinvMultiplier.h
    MatrixWVinvMultiplier.hpp
    MatrixWtXMultiplier.h
    MatrixWtXMultiplier.hpp
    GaussNewton.h
    GaussNewton.hpp
    SolverCholeskyDense.h
    SolverCholeskyDense.hpp
    SolverCholeskySchur.h
    SolverCholeskySchur.hpp
    SolverCholeskySparse.h
    SolverCholeskySparse.hpp
    SolverDiagonalBlocksInverseSchur.h
    SolverDiagonalBlocksInverseSchur.hpp
    SolverDiagonalBlocksInverseWWtMultSchur.h
    SolverDiagonalBlocksInverseWWtMultSchur.hpp
    SolverTraitsBase.h
    SolverTraits.h
    ParallelExecHelper.h
    LevenbergMarquardt.h
    LevenbergMarquardt.hpp
    OrderingTraits.h
	PRIVATE
    Global.cpp
    BaseSection.cpp
    SingleSectionHessian.cpp
    DoubleSectionHessian.cpp
    SingleSection.cpp
    DoubleSection.cpp
    EdgeSectionBase.cpp
    EdgeInterface.cpp
    EdgeUnary.cpp
    EdgeNary.cpp
    EdgeBinary.cpp
    MatrixDenseWrapper.cpp
    MatrixSparseWrapper.cpp
    MatrixDiagInv.cpp
    MatrixWWtMultiplier.cpp
    MatrixWVinvMultiplier.cpp
    MatrixWtXMultiplier.cpp
    GaussNewton.cpp
    SolverCholeskyDense.cpp
    SolverCholeskySchur.cpp
    SolverCholeskySparse.cpp
    SolverDiagonalBlocksInverseSchur.cpp
    SolverDiagonalBlocksInverseWWtMultSchur.cpp
    ParallelExecHelper.cpp
    LevenbergMarquardt.cpp
)

# if you do not want to have nelsond.{dll,lib,so} in debug disable this 
set_target_properties(nelson  PROPERTIES DEBUG_POSTFIX "d")

# special compiler flags for this target
# PUBLIC will propagate do dependent, PRIVATE if they are only for this project
#if(MSVC)
#	target_compile_options(nelson <<PUBLIC|PRIVATE>> /bigobj)
#endif()
#target_compile_features(nelson PUBLIC cxx_std_11) # cxx_std_14

#Add an alias so that library can be used inside the build tree
add_library(nelson::nelson ALIAS nelson)

# link libraries
target_link_libraries(nelson
	PUBLIC
    mat::mat
		Eigen3::Eigen
    OpenMP::OpenMP_CXX
	PRIVATE
)

# add qt automoc per target if you need (or remove this section if qt independent)
# AUTOMOC for signal slot, AUTOUIC for User Interfaces, AUTORCC for resource files
# set_target_properties(nelson PROPERTIES AUTOMOC ON)
# set_target_properties(nelson PROPERTIES AUTOUIC ON)
# set_target_properties(nelson PROPERTIES AUTORCC ON)

#------------------------------------------------------------------------------------------
# GLOBAL, SELF or a custom export name allowed
installAllTargets(SELF)

#------------------------------------------------------------------------------------------
# this will add the samples and tests folder (if existing) and will perfom common operations:
# - link the target to the nelsonVersion library, which allow to have the root include directory added
# - create folder structure for MSVC IDE
# - analyze Dll dependencies per target and globally, which in turn will be used for deep installation and PATH generation
commonEndProjectCMake(nelsonSol)
