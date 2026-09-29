vcpkg_download_distfile(archive
    URLS "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntimegenai.winml/0.15.2/microsoft.ml.onnxruntimegenai.winml.0.15.2.nupkg"
    FILENAME "microsoft.ml.onnxruntimegenai.winml.0.15.2.nupkg"
    SHA512 fd6c9d06aba7d16906ce65370d0009d549e038289ffe9ca8e73b60568937c739ccc52cbf81938ef1e92cf24b6b93efcddf382986fe6d7ef84dfb4e2aba36f883)
vcpkg_extract_source_archive(source_path ARCHIVE "${archive}" NO_REMOVE_ONE_LEVEL)

file(INSTALL "${source_path}/build/native/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
foreach(configuration IN ITEMS "" debug/)
    file(INSTALL "${source_path}/runtimes/win-x64/native/onnxruntime-genai.lib"
        DESTINATION "${CURRENT_PACKAGES_DIR}/${configuration}lib")
    file(INSTALL "${source_path}/runtimes/win-x64/native/onnxruntime-genai.dll"
        DESTINATION "${CURRENT_PACKAGES_DIR}/${configuration}bin")
endforeach()

file(WRITE "${CURRENT_PACKAGES_DIR}/share/onnxruntime-genai-winml/onnxruntime-genai-winml-config.cmake" [=[
include(CMakeFindDependencyMacro)
find_dependency(microsoft.windows.ai.machinelearning CONFIG)
if(NOT TARGET ONNXRuntimeGenAI::ONNXRuntimeGenAI)
    get_filename_component(_ort_genai_prefix "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    add_library(ONNXRuntimeGenAI::ONNXRuntimeGenAI SHARED IMPORTED)
    set_target_properties(ONNXRuntimeGenAI::ONNXRuntimeGenAI PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ort_genai_prefix}/include"
        INTERFACE_LINK_LIBRARIES "WindowsML::OnnxRuntime"
        IMPORTED_IMPLIB "${_ort_genai_prefix}/lib/onnxruntime-genai.lib"
        IMPORTED_LOCATION "${_ort_genai_prefix}/bin/onnxruntime-genai.dll"
        IMPORTED_IMPLIB_DEBUG "${_ort_genai_prefix}/debug/lib/onnxruntime-genai.lib"
        IMPORTED_LOCATION_DEBUG "${_ort_genai_prefix}/debug/bin/onnxruntime-genai.dll")
    unset(_ort_genai_prefix)
endif()
]=])

vcpkg_install_copyright(FILE_LIST
    "${source_path}/LICENSE"
    "${source_path}/ThirdPartyNotices.txt")
