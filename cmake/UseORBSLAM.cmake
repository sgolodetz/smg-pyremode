####################
# UseORBSLAM.cmake #
####################

SET(ORBSLAM_ROOT_DIR "${PROJECT_SOURCE_DIR}/../orbslam2" CACHE PATH "The ORB-SLAM root directory")
SET(Pangolin_ROOT_DIR "${ORBSLAM_ROOT_DIR}/Thirdparty/Pangolin" CACHE PATH "The Pangolin root directory")
SET(Pangolin_LIB_DIR "${Pangolin_ROOT_DIR}/install/lib")

SET(ORBSLAM_INCLUDE_DIRS
"${ORBSLAM_ROOT_DIR}"
"${ORBSLAM_ROOT_DIR}/include"
)

SET(Pangolin_INCLUDE_DIRS
"${Pangolin_ROOT_DIR}/include"
"${Pangolin_ROOT_DIR}/build/src/include"
"${Pangolin_ROOT_DIR}/install/include"
)

INCLUDE_DIRECTORIES(${ORBSLAM_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})

FIND_LIBRARY(ORBSLAM_LIBRARY_DEBUG ORB_SLAM2 HINTS "${ORBSLAM_ROOT_DIR}/build/Debug")
FIND_LIBRARY(DBoW2_LIBRARY_DEBUG DBoW2 HINTS "${ORBSLAM_ROOT_DIR}/Thirdparty/DBoW2/lib/Debug")
FIND_LIBRARY(g2o_LIBRARY_DEBUG g2o HINTS "${ORBSLAM_ROOT_DIR}/Thirdparty/g2o/bin/Debug")
FIND_LIBRARY(Pangolin_LIBRARY_DEBUG pangolin HINTS "${Pangolin_ROOT_DIR}/build/src/Debug")
FIND_LIBRARY(Pangolin_GLEW_LIBRARY_DEBUG glewd HINTS "${Pangolin_LIB_DIR}")
FIND_LIBRARY(Pangolin_PNG_LIBRARY_DEBUG libpng16_staticd HINTS "${Pangolin_LIB_DIR}")
FIND_LIBRARY(Pangolin_ZLIB_LIBRARY_DEBUG zlibstaticd HINTS "${Pangolin_LIB_DIR}")

FIND_LIBRARY(ORBSLAM_LIBRARY_RELEASE ORB_SLAM2 HINTS "${ORBSLAM_ROOT_DIR}/build/Release")
FIND_LIBRARY(DBoW2_LIBRARY_RELEASE DBoW2 HINTS "${ORBSLAM_ROOT_DIR}/Thirdparty/DBoW2/lib/Release")
FIND_LIBRARY(g2o_LIBRARY_RELEASE g2o HINTS "${ORBSLAM_ROOT_DIR}/Thirdparty/g2o/bin/Release")
FIND_LIBRARY(Pangolin_LIBRARY_RELEASE pangolin HINTS "${Pangolin_ROOT_DIR}/build/src/Release")
FIND_LIBRARY(Pangolin_GLEW_LIBRARY_RELEASE glew HINTS "${Pangolin_LIB_DIR}")
FIND_LIBRARY(Pangolin_PNG_LIBRARY_RELEASE libpng16_static HINTS "${Pangolin_LIB_DIR}")
FIND_LIBRARY(Pangolin_ZLIB_LIBRARY_RELEASE zlibstatic HINTS "${Pangolin_LIB_DIR}")

SET(ORBSLAM_LIBRARY debug ${ORBSLAM_LIBRARY_DEBUG} optimized ${ORBSLAM_LIBRARY_RELEASE})
SET(DBoW2_LIBRARY debug ${DBoW2_LIBRARY_DEBUG} optimized ${DBoW2_LIBRARY_RELEASE})
SET(g2o_LIBRARY debug ${g2o_LIBRARY_DEBUG} optimized ${g2o_LIBRARY_RELEASE})
SET(Pangolin_LIBRARY debug ${Pangolin_LIBRARY_DEBUG} optimized ${Pangolin_LIBRARY_RELEASE})
SET(Pangolin_GLEW_LIBRARY debug ${Pangolin_GLEW_LIBRARY_DEBUG} optimized ${Pangolin_GLEW_LIBRARY_RELEASE})
SET(Pangolin_PNG_LIBRARY debug ${Pangolin_PNG_LIBRARY_DEBUG} optimized ${Pangolin_PNG_LIBRARY_RELEASE})
SET(Pangolin_ZLIB_LIBRARY debug ${Pangolin_ZLIB_LIBRARY_DEBUG} optimized ${Pangolin_ZLIB_LIBRARY_RELEASE})
FIND_LIBRARY(Pangolin_JPEG_LIBRARY jpeg HINTS "${Pangolin_LIB_DIR}")

SET(ORBSLAM_LIBRARIES
${ORBSLAM_LIBRARY}
${DBoW2_LIBRARY}
${g2o_LIBRARY}
${Pangolin_LIBRARY}
${Pangolin_GLEW_LIBRARY}
${Pangolin_JPEG_LIBRARY}
${Pangolin_PNG_LIBRARY}
${Pangolin_ZLIB_LIBRARY}
)

# Search for Pangolin dependencies and automatically use them if found - this mirrors what is done by Pangolin.
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Pangolin_ROOT_DIR}/CMakeModules")

FIND_PACKAGE(MediaFoundation)

FIND_PACKAGE(TIFF)
IF(TIFF_FOUND)
  INCLUDE_DIRECTORIES(${TIFF_INCLUDE_DIR})
ENDIF()

FIND_PACKAGE(zstd)
IF(zstd_FOUND)
  INCLUDE_DIRECTORIES(${zstd_INCLUDE_DIR})
ENDIF()
