vcpkg_download_distfile(archive
    URLS "https://api.nuget.org/v3-flatcontainer/microsoft.windows.ai.machinelearning/2.3.42/microsoft.windows.ai.machinelearning.2.3.42.nupkg"
    FILENAME "microsoft.windows.ai.machinelearning.2.3.42.nupkg"
    SHA512 3f31e022d01539ec9e62d1091b484901ab67a806a2437b0f1ddfd95424abc7526092f346b52c1cd9e5465a1b6292207cff6c53f3c468b708dffb1155ade7d341)
vcpkg_extract_source_archive(source_path ARCHIVE "${archive}" NO_REMOVE_ONE_LEVEL)

file(INSTALL "${source_path}/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
foreach(configuration IN ITEMS "" debug/)
    file(INSTALL
        "${source_path}/lib/native/x64/Microsoft.Windows.AI.MachineLearning.lib"
        "${source_path}/lib/native/x64/onnxruntime.lib"
        DESTINATION "${CURRENT_PACKAGES_DIR}/${configuration}lib")
    file(INSTALL
        "${source_path}/runtimes/win-x64/native/Microsoft.Windows.AI.MachineLearning.dll"
        "${source_path}/runtimes/win-x64/native/onnxruntime.dll"
        "${source_path}/runtimes/win-x64/native/DirectML.dll"
        DESTINATION "${CURRENT_PACKAGES_DIR}/${configuration}bin")
endforeach()

file(WRITE "${CURRENT_PACKAGES_DIR}/share/microsoft.windows.ai.machinelearning/microsoft.windows.ai.machinelearning-config.cmake" [=[
if(NOT TARGET WindowsML::Api)
    get_filename_component(_windows_ml_prefix "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    add_library(WindowsML::Api SHARED IMPORTED)
    set_target_properties(WindowsML::Api PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_windows_ml_prefix}/include"
        IMPORTED_IMPLIB "${_windows_ml_prefix}/lib/Microsoft.Windows.AI.MachineLearning.lib"
        IMPORTED_LOCATION "${_windows_ml_prefix}/bin/Microsoft.Windows.AI.MachineLearning.dll"
        IMPORTED_IMPLIB_DEBUG "${_windows_ml_prefix}/debug/lib/Microsoft.Windows.AI.MachineLearning.lib"
        IMPORTED_LOCATION_DEBUG "${_windows_ml_prefix}/debug/bin/Microsoft.Windows.AI.MachineLearning.dll")
    add_library(WindowsML::OnnxRuntime SHARED IMPORTED)
    set_target_properties(WindowsML::OnnxRuntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_windows_ml_prefix}/include/winml"
        IMPORTED_IMPLIB "${_windows_ml_prefix}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${_windows_ml_prefix}/bin/onnxruntime.dll"
        IMPORTED_IMPLIB_DEBUG "${_windows_ml_prefix}/debug/lib/onnxruntime.lib"
        IMPORTED_LOCATION_DEBUG "${_windows_ml_prefix}/debug/bin/onnxruntime.dll")
    add_library(WindowsML::WindowsML INTERFACE IMPORTED)
    set_target_properties(WindowsML::WindowsML PROPERTIES
        INTERFACE_LINK_LIBRARIES "WindowsML::Api;WindowsML::OnnxRuntime")
    unset(_windows_ml_prefix)
endif()
]=])

vcpkg_install_copyright(FILE_LIST
    "${source_path}/license.txt"
    "${source_path}/ThirdPartyNotices.txt")
