CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
FILE(TO_CMAKE_PATH "$ENV{ICG_CMAKE_MODULE_PATH}" OT_ICG_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH} ${OT_ICG_CMAKE_MODULE_PATH})


PROJECT(ICGRF-houghobject)

SET( RUN_TEST FALSE )

SET ( DEBUG_MODE ON )

SET ( CMAKE_ECLIPSE_MAKE_ARGUMENTS "${CMAKE_ECLIPSE_MAKE_ARGUMENTS} -j8" )

SET ( CMAKE_CXX_FLAGS "-g" )



# check out the current host (code by Paul W.)
find_program(HOSTNAME_CMD NAMES hostname)
exec_program(${HOSTNAME_CMD} ARGS OUTPUT_VARIABLE HOSTNAME)
if (${HOSTNAME} MATCHES "graceland*")
	MESSAGE("Building on one of the gracelands")
	SET(ONGRACELANDS TRUE)
else (${HOSTNAME} MATCHES "graceland*")
	MESSAGE("Building at home")
	SET(ONGRACELANDS FALSE)
endif (${HOSTNAME} MATCHES "graceland*")


# open cv
IF (ONGRACELANDS)
  SET(OpenCV_INCLUDE_DIR /mount/nas5/schulter/opencv2/opencv_2.4.6.1_installation/usr/local/include)
  SET(OpenCV_LINK_DIRECTORIES /mount/nas5/schulter/opencv2/opencv_2.4.6.1_installation/usr/local/lib)
  SET(OpenCV_LIBRARIES opencv_core
    opencv_features2d
    opencv_contrib
    opencv_highgui
    opencv_imgproc
    opencv_legacy
    opencv_ml
    opencv_nonfree
    opencv_ts)
  SET(OpenCV_FOUND TRUE)
ELSE (ONGRACELANDS)
  FIND_PACKAGE(OpenCV REQUIRED)
ENDIF (ONGRACELANDS)

IF (OpenCV_FOUND)
  MESSAGE(STATUS "OpenCV was found.")
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
ELSE (OpenCV_FOUND)
  MESSAGE(FATAL_ERROR "OpenCV was not found.")
ENDIF (OpenCV_FOUND)



# Boost ===================================================================
IF (ONGRACELANDS)
  SET(Boost_INCLUDE_DIR /home/staff/schulter/AA_cpp_libs/boost)
  SET(Boost_LIBRARY_DIRS /home/staff/schulter/AA_cpp_libs/boost)
  SET(Boost_FOUND TRUE)
ELSE (ONGRACELANDS)
  FIND_PACKAGE(Boost 1.36 COMPONENTS program_options REQUIRED)
ENDIF (ONGRACELANDS)
IF (Boost_FOUND)
  MESSAGE(STATUS "Boost was found.")
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ELSE (Boost_FOUND)
  MESSAGE(FATAL_ERROR "Boost was not found.")
ENDIF (Boost_FOUND)

# OpenMP
FIND_PACKAGE(OpenMP)
IF (OPENMP_FOUND)
  MESSAGE(STATUS "OpenMP was found.")
ELSE (OPENMP_FOUND)
  MESSAGE(FATAL_ERROR "OpenMP was not found.")
ENDIF (OPENMP_FOUND)


# Custom/Other Stuff ======================================================
IF (ONGRACELANDS)
  SET(CUSTOM_LOCAL_INCLUDE /usr/local/include 
                           /home/staff/schulter/AA_cpp_libs 
                           /home/staff/schulter/AA_cpp_libs/libconfigpp 
                           /mount/nas1/schulter/qt/qt_build/include)
  SET(CUSTOM_LOCAL_LIB_DIRS /usr/local/lib 
                            /home/staff/schulter/AA_cpp_libs/libconfigpp 
                            /mount/nas1/schulter/qt/qt_build/lib)
ELSE (ONGRACELANDS)
  SET(CUSTOM_LOCAL_INCLUDE /usr/local/include)
  SET(CUSTOM_LOCAL_LIB_DIRS /usr/local/lib)
ENDIF (ONGRACELANDS)



# PLATFORM SPECIFIC STUFF
IF(WIN32)

  # TBA

ELSE(WIN32)
  ADD_DEFINITIONS(-O3 -mtune=native -msse3 -mfpmath=sse -fopenmp ${OpenMP_CXX_FLAGS})

  INCLUDE_DIRECTORIES (
    ${CUSTOM_LOCAL_INCLUDE}
    ${OPENNI_INCLUDES}
  )

  LINK_DIRECTORIES (
    ${CUSTOM_LOCAL_LIB_DIRS}
    #${Boost_LIBRARY_DIRS}
    ${OpenCV_LINK_DIRECTORIES}
  )

ENDIF(WIN32)


# SOURCE FILES
FILE ( GLOB APP_HEADERS *.h )
FILE ( GLOB APP_SOURCES *.cpp )
FILE ( GLOB LIB_HEADERS ../../rflib/*.h )
FILE ( GLOB LIB_SOURCES ../../rflib/*.cpp )
FILE ( GLOB LIBCORE_HEADERS ../../rflib/core/*.h )
FILE ( GLOB LIBCORE_SOURCES ../../rflib/core/*.cpp )

# INCLUDE THE CORE HEADERS
INCLUDE_DIRECTORIES ( ../../rflib )
INCLUDE_DIRECTORIES ( ../../rflib/core )

# CORE LIBRARY DECLARATION
ADD_LIBRARY ( icg_rf_core  
	${LIB_HEADERS}
	${LIB_SOURCES}
	${LIBCORE_HEADERS}
	${LIBCORE_SOURCES}
)

# ADD THE EXECUTABLE
ADD_EXECUTABLE ( icgrf-hough 
	${APP_HEADERS}
	${APP_SOURCES}
)





IF(WIN32)

  # TBA

ELSE(WIN32)
  TARGET_LINK_LIBRARIES(icgrf-hough
    icg_rf_core
    config++
    gomp
    #${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )
ENDIF(WIN32)


