{ cmake
, tbb
, range-v3
, tinyobjloader
, yaml-cpp
, lyra
, eigen
, python3
, pystring
, meson-patched
, pkg-config
, ninja
, spdlog
, nlohmann_json
, highfive
, fast-cpp-csv-parser
, fmt
, dhall-json
, cudaPackages_12_2
, llvmPackages_16
, mkShell
, clang-tools
, haskellPackages
, ffmpeg
, cuda-samples
, hdf5
, cnpy
, happly
, hougeo
, amgcl
, cereal
}:
let
  clang-tools_ = clang-tools.override { llvmPackages = llvmPackages_16; };
in
mkShell {
  shellHook = ''
    export PATH=${clang-tools_}/bin:$PATH
    export LD_LIBRARY_PATH=/run/opengl-driver/lib
    export NIX_CFLAGS_COMPILE=" -isystem ${eigen}/include/eigen3$NIX_CFLAGS_COMPILE"
  '';
  nativeBuildInputs = [
    cmake
    meson-patched
    pkg-config
    ninja
    (haskellPackages.ghcWithPackages (ps: with ps;[
      shake
      utf8-string
      aeson
      raw-strings-qq
      haskell-language-server
    ]))
  ];
  buildInputs = [
    ffmpeg
    tbb
    range-v3
    tinyobjloader
    yaml-cpp
    lyra
    eigen
    (python3.withPackages (ps: with ps; [
      pybind11
      numpy
    ]))
    spdlog
    nlohmann_json
    pystring
    cudaPackages_12_2.cudatoolkit
    highfive
    fmt
    fast-cpp-csv-parser
    dhall-json
    cuda-samples
    hdf5
    cnpy
    happly
    hougeo
    amgcl
    llvmPackages_16.openmp
    cereal
  ];
}
