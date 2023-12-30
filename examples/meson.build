executable(
  'example',
  sources: 'main.cu',
  link_with: cuNSearch,
  include_directories: inc
)

executable(
  'test-template',
  sources: 'test.cu',
  include_directories: [ inc, '../Utils' ],
  override_options: ['cpp_std=c++17', 'cuda_std=c++17'],
  cuda_args : [
    '--extended-lambda',
    '--expt-relaxed-constexpr',
  ],
  link_with: cuNSearch,
  dependencies: [
    fmt,
    spdlog,
  ],
)
