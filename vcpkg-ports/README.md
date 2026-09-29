# Inference dependency overlay

The `ryzen-ai` feature installs the official Windows ML and ONNX Runtime GenAI
native NuGet packages through pinned vcpkg ports. Headers, import libraries,
runtime DLLs and their license files all come from those packages.

The AMD VitisAI execution provider is acquired on the target machine through
the Windows ML execution-provider catalog.
