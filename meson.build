project('cuNSearch', 'cpp', 'cuda')
add_global_arguments('-DCUNSEARCH_USE_DOUBLE_PRECISION', language : 'cuda')

inc = include_directories('include')

cuNSearch = library(
  'cuNSearch',
  [
    'src/PointSet.cu',
    'src/PointSetImplementation.cu',
    'src/cuNSearch.cu',
    'src/cuNSearchDeviceData.cu',
    'src/cuNSearchKernels.cu',
    'Utils/cuda_helper.cu',
    'Utils/Timing.cpp',
    'Utils/IDFactory.cpp',
  ],
  include_directories: [
    inc,
    'Utils',
  ],
)

subdir('examples')
