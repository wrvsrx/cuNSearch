project('cuNSearch', 'cpp', 'cuda')

set (INCLUDE_HEADERS
	include/PointSet.h
	include/ActivationTable.h
	include/Common.h
	include/cuNSearch.h
)

set (HEADER_FILES
	src/Types.h
	src/cuNSearchDeviceData.h
	src/GridInfo.h
	src/NotImplementedException.h
	src/PointSetImplementation.h
	src/cuNSearchKernels.cuh
	src/helper_linearIndex.h
	src/helper_mortonCode.h
	Utils/cuda_helper.h
	Utils/Timing.h
	Utils/IDFactory.h
)

set (SOURCE_FILES
	src/PointSet.cu
	src/PointSetImplementation.cu
	src/cuNSearch.cu
	src/cuNSearchDeviceData.cu
	src/cuNSearchKernels.cu
	Utils/cuda_helper.cu
	Utils/Timing.cpp
	Utils/IDFactory.cpp
)

if(BUILD_AS_SHARED_LIBS)
	add_library(cuNSearch SHARED ${INCLUDE_HEADERS} ${HEADER_FILES} ${SOURCE_FILES})
else()
	add_library(cuNSearch STATIC ${INCLUDE_HEADERS} ${HEADER_FILES} ${SOURCE_FILES})
endif()

include(GenerateExportHeader)
generate_export_header(cuNSearch
	BASE_NAME cuNSearch
	EXPORT_MACRO_NAME cuNSearch_EXPORT
	EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/cuNSearch/cuNSearch_export.h
	)

target_include_directories(cuNSearch PUBLIC
	"include"
	"Utils"
	${CUDA_INCLUDE_DIRS}
	${CMAKE_BINARY_DIR}/cuNSearch
)
target_link_libraries(cuNSearch PUBLIC ${CUDA_LIBRARIES})
target_compile_definitions(cuNSearch PUBLIC $<$<CONFIG:DEBUG>:DEBUG>)

list(APPEND INCLUDE_HEADERS
			${CMAKE_CURRENT_BINARY_DIR}/cuNSearch/cuNSearch_export.h)

install(FILES ${INCLUDE_HEADERS}
	DESTINATION include/)

install(TARGETS cuNSearch
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

option(BUILD_DEMO "Build example of how to use this library."
		ON)
if(BUILD_DEMO)
	add_subdirectory(demo)
endif(BUILD_DEMO)


unset(USE_DOUBLE_PRECISION CACHE)
