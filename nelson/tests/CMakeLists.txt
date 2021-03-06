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
  testMatrixWWtMultiplier.cpp
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
OPTION(nelsonTest_COMPILE_LONG_TESTS "Compile All Tests, including very long compile time ones" ON)

if(nelsonTest_COMPILE_LONG_TESTS)
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


#--------------------------------------------------------------------------------------------


  set(DOUBLESECTIONSCHURTESTCOMMON testDoubleSectionSchurCommon.h testDoubleSectionSchurCommon.hpp)


  set(TESTS_LIST 
    testDoubleSectionSchurDiagBlockFFFF.cpp
    testDoubleSectionSchurDiagBlockFFFD.cpp
    testDoubleSectionSchurDiagBlockFFDF.cpp
    testDoubleSectionSchurDiagBlockFFDD.cpp
    #--------------------------------------
    testDoubleSectionSchurDiagBlockFDFF.cpp
    testDoubleSectionSchurDiagBlockFDFD.cpp
    testDoubleSectionSchurDiagBlockFDDF.cpp
    testDoubleSectionSchurDiagBlockFDDD.cpp
    #--------------------------------------
    testDoubleSectionSchurDiagBlockFVFF.cpp
    testDoubleSectionSchurDiagBlockFVFD.cpp
    testDoubleSectionSchurDiagBlockFVDF.cpp
    testDoubleSectionSchurDiagBlockFVDD.cpp
    #--------------------------------------
    #--------------------------------------
    testDoubleSectionSchurDiagBlockDFFF.cpp
    testDoubleSectionSchurDiagBlockDFFD.cpp
    testDoubleSectionSchurDiagBlockDFDF.cpp
    testDoubleSectionSchurDiagBlockDFDD.cpp
    #--------------------------------------
    testDoubleSectionSchurDiagBlockDDFF.cpp
    testDoubleSectionSchurDiagBlockDDFD.cpp
    testDoubleSectionSchurDiagBlockDDDF.cpp
    testDoubleSectionSchurDiagBlockDDDD.cpp
    #--------------------------------------
    testDoubleSectionSchurDiagBlockDVFF.cpp
    testDoubleSectionSchurDiagBlockDVFD.cpp
    testDoubleSectionSchurDiagBlockDVDF.cpp
    testDoubleSectionSchurDiagBlockDVDD.cpp
    #--------------------------------------
    #--------------------------------------
    testDoubleSectionSchurDiagBlockVFFF.cpp
    testDoubleSectionSchurDiagBlockVFFD.cpp
    testDoubleSectionSchurDiagBlockVFDF.cpp
    testDoubleSectionSchurDiagBlockVFDD.cpp
    #--------------------------------------
    testDoubleSectionSchurDiagBlockVDFF.cpp
    testDoubleSectionSchurDiagBlockVDFD.cpp
    testDoubleSectionSchurDiagBlockVDDF.cpp
    testDoubleSectionSchurDiagBlockVDDD.cpp
    #--------------------------------------
    testDoubleSectionSchurDiagBlockVVFF.cpp
    testDoubleSectionSchurDiagBlockVVFD.cpp
    testDoubleSectionSchurDiagBlockVVDF.cpp
    testDoubleSectionSchurDiagBlockVVDD.cpp
  )

  foreach(s ${TESTS_LIST})
    get_filename_component(sn "${s}" NAME_WE)
    add_executable(nelsonTest_${sn})
    target_sources_wrapper(nelsonTest_${sn} PRIVATE ${s} ${DOUBLESECTIONSCHURTESTCOMMON})
    if (WIN32)
      set_target_properties(nelsonTest_${sn} PROPERTIES FOLDER nelson/test)
    endif()	
    if(MSVC)
      target_compile_options(nelsonTest_${sn} PRIVATE /bigobj)
    endif()
    target_link_libraries(nelsonTest_${sn} PRIVATE nelson::nelson Catch2::Catch2WithMain)
    
    # add to tests
    catch_discover_tests(nelsonTest_${sn})
  endforeach()
endif()

######################################################################

find_package_wrapper(lie REQUIRED CONFIG)

set(TESTS_LIST 
  testGaussNewton2DScansAlign.cpp
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
