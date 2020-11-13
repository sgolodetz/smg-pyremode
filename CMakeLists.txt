cmake_minimum_required(VERSION 3.10)

PROJECT(pyremode LANGUAGES CXX)

FIND_PACKAGE(pybind11 CONFIG REQUIRED)

###########################
# Specify the target name #
###########################

SET(targetname pyremode)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseEigen.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenCV.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseREMODE.cmake)

##########################################
# Specify additional include directories #
##########################################

INCLUDE_DIRECTORIES(${pybind11_INCLUDE_DIRS})

######################
# Specify the target #
######################

ADD_LIBRARY(${targetname} SHARED ${targetname}.cpp ${targetname}.pyi)
SET_TARGET_PROPERTIES(${targetname} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")

#################################
# Specify the libraries to link #
#################################

TARGET_LINK_LIBRARIES(${targetname} debug pybind11::module optimized pybind11::module)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkOpenCV.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkREMODE.cmake)

##########################################
# Copy resource files to the right place #
##########################################

ADD_CUSTOM_COMMAND(TARGET ${targetname} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PROJECT_SOURCE_DIR}/${targetname}.pyi" "$<TARGET_FILE_DIR:${targetname}>")
