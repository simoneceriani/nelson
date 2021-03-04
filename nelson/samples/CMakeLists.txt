cmake_minimum_required(VERSION 3.17)
# note, suggested name is PARENT_FOLDER_samples (e.g., parent folder is myLib, this is myLib_samples
PROJECT(nelson_samples LANGUAGES CXX) 

## complex case, create a dir and do it as a standard project
#add_subdirectory(...)

# note: samples are not installed by default, 
# we do not use target_sources_wrapper here 
# but the native target_sources function

## add these lines block for each sample, or if you need more complex structure, create a subfolder
add_executable(nelsonSamples_WWtPerformances)
target_sources(
  nelsonSamples_WWtPerformances 
  PRIVATE 
  WWtMultPerformances.cpp 
  Problems.h 
  Problems.cpp
)
if (WIN32)
 	set_target_properties(nelsonSamples_WWtPerformances PROPERTIES FOLDER nelson/samples)
endif()
target_link_libraries(nelsonSamples_WWtPerformances PRIVATE nelson)

## and/or compile this list to have simple 1-file samples

set(SAMPLES_LIST 
	nelsonPlayground.cpp  
)

foreach(s ${SAMPLES_LIST})
	get_filename_component(sn "${s}" NAME_WE)
	add_executable(nelsonSamples_${sn})
	target_sources(nelsonSamples_${sn} PRIVATE ${s})
	if (WIN32)
		set_target_properties(nelsonSamples_${sn} PROPERTIES FOLDER nelson/samples)
	endif()	
	target_link_libraries(nelsonSamples_${sn} PRIVATE nelson)
endforeach()