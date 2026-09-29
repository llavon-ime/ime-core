# IME Core

Llavon IME 的跨平台靜態 C++ 推論函式庫。本專案負責模型載入、斷詞、候選遮蔽
（candidate masking）、llama.cpp 推論與每位客戶端的推論 session；不含服務
IPC、處理程序啟動或作業系統特定的路徑探索。

`EncodingTables` 可在不載入模型的情況下，將推論用分詞器與注音候選字表
提供給離線資料集建置工具使用。

## 建置

模型載入時會一次預備 Vulkan 管線。預備範圍包含小型推論批次與已有內容的
KV 快取；完成後會清除合成狀態，並將預備好的推論內容交給第一個工作階段使用。
如此可避免把耗時的管線建立延後到最初幾次按鍵輸入。這會增加模型載入
時間，但不會在閒置期間定期執行。

當主程式提供 `CoreConfig::vulkan_pipeline_cache_dir` 時，核心會在載入模型前設定
Vulkan 後端，並於預備完成後儲存管線快取。路徑為空時會停用持久化快取。
若後端未提供這個選用的函式查詢擴充，核心仍可正常運作。

設定 `IME_CORE_BUILD_TOOLS=ON` 可建置 `ime-core-latency-check`。執行時依序傳入
`MODEL_PATH TABLES_DIRECTORY VULKAN_DEVICE_ID`，例如 `Vulkan0`。另外傳入
`--cache-dir DIRECTORY` 才會啟用持久化管線快取。此工具會測試兩個全新
工作階段、不同的上下文長度、250 毫秒間隔，以及閒置 2.3 秒後恢復推論；包含
第一次在內的每項請求都會輸出結果，並列出候選字，供修改前後比對正確性。
加上 `--load-only` 可在全新處理程序中只診斷模型載入與快取。

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
