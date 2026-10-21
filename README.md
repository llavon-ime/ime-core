# ime-core

拉風輸入法的跨平台推論核心，以 C++23 靜態函式庫形式提供。

給定前文與注音序列，ime-core 會回傳每個位置的候選字及其分數，而且只會輸出符合該讀音的字。
模型載入、斷詞、候選字遮蔽與推論都在這裡完成；IPC、程序管理、設定檔與平台相關的路徑都交給
呼叫端處理，因此 Windows、macOS 與 Linux 版的輸入法共用同一份核心。

## 職責範圍

| 由 ime-core 負責 | 由呼叫端負責 |
| --- | --- |
| 載入 GGUF 模型、選擇推論裝置 | 決定模型、字表與快取放在哪裡 |
| 斷詞與注音候選字表 | 讀寫應用程式設定 |
| 依讀音遮蔽候選字並排序 | IPC、程序生命週期 |
| 管理每位用戶端的推論 session | 記錄的傳輸、排隊與顯示 |

## 使用方式

標頭檔位於 `include/ime-core/`，所有型別都在 `llavon::ime::core` 命名空間下。

```cpp
#include <ime-core/core.hpp>

namespace core = llavon::ime::core;

core::Core engine{core::CoreConfig{
    .model_path = "llavon-ime-llama-250m-Q4_K_M.gguf",
    .tables_dir = "share/ime-core/tables",
    .logger = std::make_shared<MyLogger>(),  // 必填，實作 core::Logger
}};

auto session = engine.create_session();
session->ready();  // 閒置時預熱，可省略
auto predictions = session->predict(context, padding);
```

- **`Core`**：持有已載入的模型。建構時就會載入模型；設定無效或載入失敗時拋出例外。
- **`Session`**：由 `Core::create_session()` 建立，每位用戶端各一個，保有該用戶端的推論 context。
  `predict()` 回傳的 `Prediction` 與 `padding` 一一對應，每個位置都是一串 `(字, 分數)`。
- **`PaddingEntry`**：表示一個待預測的位置，內含該位置的注音；若使用者已手動選字，
  則設定 `chosen` 與 `chosen_char`。
- **`EncodingTables`**（`encoding_tables.hpp`）：不載入模型，只提供與推論相同的斷詞與候選字表，
  給離線建置資料集的工具使用（例如 LoRA 訓練資料轉換）。

### `CoreConfig`

| 欄位 | 說明 |
| --- | --- |
| `model_path` | GGUF 模型路徑（必填） |
| `tables_dir` | 字表目錄（必填）；安裝後位於 `share/ime-core/tables` |
| `inference_provider` | 推論引擎；macOS 預設為 MLX，其他平台為 llama.cpp |
| `inference_device` | 推論後端與裝置 ID；預設為 `automatic` |
| `gpu_layers` | 要放到 GPU 的層數；`-2` 或 `-1` 表示全部，`0` 表示只用 CPU |
| `context_length` | context 長度；`0` 表示使用預設值 |
| `threads` | CPU 推論執行緒數，預設為 8 |
| `vulkan_pipeline_cache_dir` | Vulkan pipeline 快取目錄；留空則不使用持久快取 |
| `logger` | 記錄器（必填），詳見下方「記錄」 |

`enumerate_inference_devices()` 不需載入模型即可列出可用的推論裝置，適合用來製作設定介面。
模型載入後，可透過 `Core::inference_runtime_info()` 查詢實際使用的裝置、是否有 GPU offload，
以及是否退回 CPU。

## 記錄

ime-core 本身沒有記錄用的執行緒、佇列或傳輸通道，只透過 `CoreConfig::logger` 呼叫呼叫端提供的
`Logger`（`logger.hpp`）實作。`logger` 為必填，未提供時 `Core` 建構會拋出例外；不需要記錄時，
可傳入一個直接丟棄訊息的實作。

- `log(std::string)`：傳入已組好的訊息。
- `log(MessageFactory)`：傳入延後組字串的函式，避免在推論路徑上做昂貴的格式化。
- 兩者都有帶 `LogInformation`（`general`、`context`、`debug`）分類的多載。

實作必須遵守：不阻塞呼叫端、不拋出例外，且不得在呼叫端的執行緒上執行 `MessageFactory`。
如果記錄已停用或訊息被拒絕，`MessageFactory` 完全不能被執行；被接受的 `MessageFactory`
最多只能在記錄器的工作執行緒上執行一次。

## Vulkan 預熱與快取

使用 Vulkan 時，核心會在模型載入階段一次建好推論所需的 pipeline，涵蓋小 batch 與已有內容的
KV 快取，完成後清除這些合成狀態，並把預熱好的 context 交給第一個 session。這樣能避免最初幾次
按鍵時才建立 pipeline 而卡頓，代價是模型載入時間變長；閒置時不會定期執行任何 GPU 工作。

若有設定 `vulkan_pipeline_cache_dir`，核心會在載入模型前把快取目錄交給 Vulkan 後端，
並在預熱完成後立即儲存，之後啟動即可重複使用。後端不支援這項選用擴充時，核心仍可正常運作，
只是沒有快取。固定使用的 ggml fork 請參考 [vcpkg-ports/README.md](vcpkg-ports/README.md)。

## 建置

需要 CMake 3.24 以上、支援 C++23 的編譯器，以及 vcpkg（相依套件列於 `vcpkg.json`）。
請由呼叫端傳入 vcpkg 工具鏈，`<preset>` 依平台填入 `windows`、`linux` 或 `macos`：

```sh
cmake --preset <preset> -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build --preset <preset>
cmake --install build/<preset> --config Release
```

安裝後會匯出 CMake 套件，以 `find_package(ime-core CONFIG)` 引入，並連結靜態 target
`ime-core::ime-core`。若以子模組方式引入，也可以直接 `add_subdirectory`，不需要另外安裝。

### 建置選項

| 選項 | 說明 |
| --- | --- |
| `IME_CORE_BUILD_TESTS` | 建置單元測試，之後以 `ctest` 執行 |
| `IME_CORE_BUILD_TOOLS` | 建置需要實際模型的手動診斷工具 `ime-core-latency-check` |
| `IME_CORE_TEST_MODEL` | 指定 GGUF 模型，額外執行 MLX 後端的模型一致性測試（需有 MLX，例如在 macOS 上） |

### 延遲檢查工具

```text
ime-core-latency-check MODEL_PATH TABLES_DIRECTORY VULKAN_DEVICE_ID [--cache-dir DIRECTORY] [--load-only]
```

- `VULKAN_DEVICE_ID`：例如 `Vulkan0`。
- `--cache-dir`：啟用持久 pipeline 快取。
- `--load-only`：只量測模型載入與快取，請在新的程序中執行。

此工具會測試兩個全新的 session、不同長度的前文、間隔 250 ms 的連續請求，以及閒置 2.3 秒後
恢復輸入的情況。每個請求（包含第一個）都會輸出延遲與候選字，方便比較修改前後的效能與正確性。
