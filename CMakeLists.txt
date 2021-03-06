cmake_minimum_required(VERSION 3.17)
project(nelsonSol VERSION 1.0.0 LANGUAGES CXX)

# tell cmake where is CMakeBaseBuildSystem folder
if(nelsonSol STREQUAL ${CMAKE_PROJECT_NAME})
  set(nelsonSol_CMAKEBASEBUILDSYSTEM_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmakebasebuildsystem" CACHE PATH "path to the CMakeBaseBuildSystem folder")
  list(APPEND CMAKE_MODULE_PATH "${nelsonSol_CMAKEBASEBUILDSYSTEM_PATH}")
endif()


# add the CMakeBaseBuildSystem modules, just a bunch of includes here
include(winDllRecursiveAnalysis)
include(utils)
include(CMakeBaseBuildSystem)
include(gitTracer)

# add CMakeBaseBuildSystem in source dir, allow to track version!
if(nelsonSol_CMAKEBASEBUILDSYSTEM_PATH)
  configureGitCMakeFile(${nelsonSol_CMAKEBASEBUILDSYSTEM_PATH} CMakeBaseBuildSystem ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if (WIN32)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

	# global definitions, edit as you need, but try as much as you can to use specific target properties!
	
	# define a single output folder for binaries (dll and exe)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
	add_definitions(-DUNICODE)
	
	# silent generally an annoying warning from visual studio about dll export/import
	add_compile_options(/wd4251)
	
	# add multicore compilation in visual studio
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
	
endif()

# message on cmake_prefix_path, it is always nice to know where I am looking too!
MESSAGE(STATUS "Packages search paths:")
MESSAGE(STATUS "ENV{CMAKE_PREFIX_PATH} = $ENV{CMAKE_PREFIX_PATH}")
MESSAGE(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

# allow to use by default RELEASE lib when RELWITHDEBINFO is compiled
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
# note, if need at target level, use set_target_properties(<<target>> PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)


# enable cmake testing
enable_testing()

# add myself version, but in binary dir, otherwise each commit will dirty itself, possible future use (so far commented)
# configureGitCMakeFile(${CMAKE_CURRENT_SOURCE_DIR} nelsonSol ${CMAKE_CURRENT_BINARY_DIR})

# add fixed for libraries without modern config files if you need, 
# divide them in common and unix/win specific 
# NOTE: content of this folder will be copied at installation step
# if you need some special cmake additional file please put it somewhere else
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
#if(UNIX)
#	list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/unix")
#endif()
#if(WIN32)
#	list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/win32")
#endif()
#installSpecialCMake("${CMAKE_CURRENT_LIST_DIR}/cmake/")


# set default option values
# set ON if your project is an application and need to be standalone, OFF if is a library to be distributed
OPTION(nelsonSol_INSTALL_DEPENDENCIES "Install Dll dependencies" OFF)
OPTION(nelsonSol_BUILD_SAMPLES "Global switch for samples building" ON)
OPTION(nelsonSol_BUILD_TESTS "Global switch for tests building" ON)

# installation destination
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installation" CACHE PATH "prefix to installation folder" FORCE)
endif()

# prepare vesion.h and version.cpp and link them in a library with name 
# NOTE: this will prepare and install a target <<MAIN_PROJECT_NAME>>Version (STATIC LIBRARY) 
prepareVersionFile(nelsonSol)
# alternative, use prepareVersionFileInterface(<<MAIN_PROJECT_NAME>>) to create <<MAIN_PROJECT_NAME>>Version as INTERFACE LIBRARY

# add the libs (please respect the structure)
OPTION(nelson_BUILD_MAT "Build mat here, if ON you need to clone mat in mat folder" OFF)
if(nelson_BUILD_MAT)
	add_subdirectory(mat)
	configureGitCMakeFile(${CMAKE_CURRENT_SOURCE_DIR}/mat mat ${CMAKE_CURRENT_SOURCE_DIR})
endif()
add_subdirectory(nelson)
# add_subdirectory(...)

#--------------------------------------------------------------------------------------
# create the config/configVersion file, 
# first argument, name of the main project 
# use additional arguments to specify defaul list of components when this will be used in FIND_PACKAGE. 
# NONE ALL or custom list are accepted
# note: if I am not the main project skip (submodule)
if(nelsonSol STREQUAL ${CMAKE_PROJECT_NAME})
	createGlobalConfigFile(nelson ALL)
endif()

# install other cross dependent dll/so if requested
if(${nelsonSol_INSTALL_DEPENDENCIES})
	prepareDLLInstall(DEBUG RELEASE)
endif()

# configure the bat file for VS, cmd with PATH, this will be produced in build folder
if(MSVC)
	prepareBatFile()
endif()

# only a print so far, TODO properly plugin management for installation!
# include(QtCMakeUtils)
# printQtPluginsList()
