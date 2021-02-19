cmake_minimum_required(VERSION 3.17)
PROJECT(nelsonTest  LANGUAGES CXX)

include(CTest)
find_package(Catch2 REQUIRED)
include(Catch)

# note: tests are not installed by default, 

# add these lines block for each test, or if you need more complex structure, create a subfolder
#add_executable(<<TARGET_NAME>>)
#target_sources_wrapper(<<TARGET_NAME>> PRIVATE <<...>>)
#if (WIN32)
#	set_target_properties(<<TARGET_NAME>> PROPERTIES FOLDER nelson/test)
#endif()
#target_link_libraries(<<TARGET_NAME>> PRIVATE <<...>> Boost::unit_test_framework)
#add_test(NAME <<TARGET_NAME>> COMMAND <<TARGET_NAME>>)

# and/or compile this list to have simple 1-file test

set(TESTS_LIST 
  testSingleSectionHessian.cpp
  testDoubleSectionHessian.cpp
  testSingleSection.cpp
  testMatrixSparseDenseWrapper.cpp
)

foreach(s ${TESTS_LIST})
	get_filename_component(sn "${s}" NAME_WE)
	add_executable(nelsonTest_${sn})
	target_sources_wrapper(nelsonTest_${sn} PRIVATE ${s})
	if (WIN32)
		set_target_properties(nelsonTest_${sn} PROPERTIES FOLDER nelson/test)
		target_compile_options(nelsonTest_${sn} PUBLIC /bigobj)
	endif()	
	target_link_libraries(nelsonTest_${sn} PRIVATE nelson::nelson Catch2::Catch2WithMain )
  
  # add to tests
  catch_discover_tests(nelsonTest_${sn})
endforeach()

######################################################################
set(DOUBLESECTIONTESTCOMMON testDoubleSectionCommon.h testDoubleSectionCommon.hpp)

set(TESTS_LIST 
  testDoublesectionFFFF.cpp
  testDoublesectionFFFD.cpp
  testDoublesectionFFDF.cpp
  testDoublesectionFFDD.cpp
  #------------------------
  testDoublesectionFDFF.cpp
  testDoublesectionFDFD.cpp
  testDoublesectionFDDF.cpp
  testDoublesectionFDDD.cpp
  #------------------------
  testDoublesectionFVFF.cpp
  testDoublesectionFVFD.cpp
  testDoublesectionFVDF.cpp
  testDoublesectionFVDD.cpp
  #------------------------
  #------------------------
  testDoublesectionDFFF.cpp
  testDoublesectionDFFD.cpp
  testDoublesectionDFDF.cpp
  testDoublesectionDFDD.cpp
  #------------------------
  testDoublesectionDDFF.cpp
  testDoublesectionDDFD.cpp
  testDoublesectionDDDF.cpp
  testDoublesectionDDDD.cpp
  #------------------------
  testDoublesectionDVFF.cpp
  testDoublesectionDVFD.cpp
  testDoublesectionDVDF.cpp
  testDoublesectionDVDD.cpp
  #------------------------
  #------------------------
  testDoublesectionVFFF.cpp
  testDoublesectionVFFD.cpp
  testDoublesectionVFDF.cpp
  testDoublesectionVFDD.cpp
  #------------------------
  testDoublesectionVDFF.cpp
  testDoublesectionVDFD.cpp
  testDoublesectionVDDF.cpp
  testDoublesectionVDDD.cpp
  #------------------------
  testDoublesectionVVFF.cpp
  testDoublesectionVVFD.cpp
  testDoublesectionVVDF.cpp
  testDoublesectionVVDD.cpp
)

foreach(s ${TESTS_LIST})
	get_filename_component(sn "${s}" NAME_WE)
	add_executable(nelsonTest_${sn})
	target_sources_wrapper(nelsonTest_${sn} PRIVATE ${s} ${DOUBLESECTIONTESTCOMMON})
	if (WIN32)
		set_target_properties(nelsonTest_${sn} PROPERTIES FOLDER nelson/test)
		target_compile_options(nelsonTest_${sn} PUBLIC /bigobj)
	endif()	
	target_link_libraries(nelsonTest_${sn} PRIVATE nelson::nelson Catch2::Catch2WithMain )
  
  # add to tests
  catch_discover_tests(nelsonTest_${sn})
endforeach()


######################################################################

find_package_wrapper(lie REQUIRED CONFIG)

set(TESTS_LIST 
  testGaussNewton2DScansAlign.cpp
  testGaussNewton2DScansAlignTimingTest.cpp
)

foreach(s ${TESTS_LIST})
	get_filename_component(sn "${s}" NAME_WE)
	add_executable(nelsonTest_${sn})
	target_sources_wrapper(nelsonTest_${sn} PRIVATE ${s})
	if (WIN32)
		set_target_properties(nelsonTest_${sn} PROPERTIES FOLDER nelson/test)
	endif()	
	if(MSVC)
		target_compile_options(nelsonTest_${sn} PRIVATE /bigobj)
	endif()
	target_link_libraries(nelsonTest_${sn} PRIVATE nelson::nelson Catch2::Catch2WithMain lie::lie)
  
  # add to tests
  catch_discover_tests(nelsonTest_${sn})
endforeach()
