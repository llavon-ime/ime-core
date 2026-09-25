# IME Core

Llavon IME 的跨平台靜態 C++ 推論函式庫。本專案負責模型載入、斷詞、候選遮蔽
（candidate masking）、llama.cpp 推論與每位客戶端的推論 session；不含服務
IPC、處理程序啟動或作業系統特定的路徑探索。

## 建置

由呼叫端傳入 vcpkg 工具鏈：

```powershell
cmake --preset windows -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build --preset windows
cmake --install build/windows --config Release
```

安裝後的 CMake 套件會匯出靜態 target `ime-core::ime-core`。

## 核心資料

`CoreConfig` 需要 GGUF 模型路徑與表目錄。安裝的參考表會放在
`share/ime-core/tables` 下。

推論裝置的選擇以資料形式透過 `CoreConfig` 提供。
`enumerate_inference_devices()` 會回報已載入的 ggml 後端所暴露的裝置，且不需
載入模型。核心不會尋找或解析應用程式的設定檔。

## 日誌

`CoreConfig::logger` 可接受平台無關的 `Logger` 介面的選用實作。
`Logger::log(std::string)` 處理已經存在的訊息，`Logger::log(MessageFactory)`
則延後昂貴的格式化。logger 絕不能在呼叫端執行緒上求值 message factory，且在
日誌停用或訊息被拒絕時完全不得求值。核心不含日誌執行緒、佇列、pipe 或平台
傳輸。
