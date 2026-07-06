# Mine Detection API

以 FastAPI 與 Ultralytics 提供地雷影像 YOLOv8 object detection 推論服務。服務只會載入既有的 `weights/best.pt`，不包含模型訓練流程。

`config/class_map.json` 是 `class_id` 到 `class_name` 的唯一對照來源。服務啟動時會驗證權重與類別表的類別數是否一致。

## 從零開始

以下流程預設使用者的電腦尚未下載本專案，也尚未建立 Python 虛擬環境。

### 1. 安裝前置工具

先安裝 [Git](https://git-scm.com/downloads)，並選擇下列其中一種執行方式：

- 本機 Python：安裝 Python 3.11 或 3.12。
- Docker：安裝 Docker Desktop 或 Docker Engine，不需要另外建立 Python 環境。

GPU 不是必要條件。沒有可用的 NVIDIA CUDA GPU 時，服務會自動使用 CPU，但推論速度會較慢。

### 2. 下載專案

```powershell
git clone https://github.com/RobertLinDoProject/mine-detection.git
cd mine-detection
```

接著確認 API 必要檔案存在：

```powershell
Test-Path .\weights\best.pt
Test-Path .\config\class_map.json
```

兩行都應顯示 `True`。`weights/best.pt` 是 API 實際使用的權重；服務不會自動下載遺失的權重。

Linux 或 macOS 可使用：

```bash
test -f weights/best.pt && echo "weights OK"
test -f config/class_map.json && echo "class map OK"
```

### 3. 安裝並啟動

選擇其中一種方式：

- 使用 Python：依照「本機執行」章節建立虛擬環境並啟動。
- 使用 Docker：依照「Docker」章節建置 image 並啟動 container。

所有指令都應在包含 `README.md`、`app/`、`config/` 與 `weights/` 的專案根目錄執行。

### 4. 確認服務可用

服務啟動後，另外開啟一個終端機：

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/metadata
```

`/health` 應回傳：

```json
{"status":"ok","model_loaded":true}
```

接著可開啟 `http://localhost:8000/docs`，直接透過 Swagger UI 測試 API，或依照下方「curl 測試」章節上傳圖片。

## 專案結構

```text
mine-detection/
├── app/
│   ├── main.py              # FastAPI 入口與 API endpoints
│   ├── detector.py          # YOLO 模型載入、裝置選擇與推論邏輯
│   └── schemas.py           # API request/response schemas
├── config/
│   └── class_map.json       # class_id 到 class_name 的唯一對照表
├── tests/
│   ├── test_api.py          # API 與圖片上傳驗收測試
│   └── test_detector.py     # 模型啟動、裝置及偵測結果測試
├── weights/
│   └── best.pt              # 推論服務使用的 YOLOv8 權重
├── Dockerfile               # Docker image 建置與服務啟動設定
├── requirements.txt         # 正式執行環境相依套件
├── requirements-dev.txt     # 測試環境額外相依套件
└── README.md                # 專案與使用說明
```

`.venv/`、`__pycache__/` 等本機產生的環境與快取目錄不屬於正式專案結構。

## API

- `GET /health`：服務及模型載入狀態
- `GET /metadata`：模型版本與類別資訊
- `POST /v1/detect`：上傳單張圖片並執行偵測
- `GET /docs`：Swagger UI

`POST /v1/detect` 支援下列 query parameters：

- `conf`：confidence threshold，預設 `0.25`，範圍 `0.0` 至 `1.0`
- `iou`：IoU threshold，預設 `0.45`，範圍 `0.0` 至 `1.0`

`detection_id` 從 `1` 開始。`bbox_normalized` 的座標範圍為 `0.0` 至 `1.0`，`latency_ms` 為模型 `predict` 的執行時間。

若模型輸出的 `class_id` 不存在於 `config/class_map.json`，該筆結果的 `class_name` 會設為 `unknown`，並在回應最上層的 `warnings` 陣列留下警告。服務不會使用權重內的類別名稱補值。

## 本機執行

建議使用 Python 3.11 或 3.12。以下命令須在專案根目錄執行。

Windows PowerShell：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

上述方式不需要執行 `Activate.ps1`，因此不會受到 PowerShell Execution Policy 阻擋。

Linux 或 macOS：

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服務啟動後可開啟 `http://localhost:8000/docs`。

服務會自動檢查 CUDA：有可用 GPU 時固定使用 `cuda:0`，只有 CUDA 不可用時才使用 CPU。可透過 `GET /metadata` 的 `device` 欄位確認目前選用裝置。

## curl 測試

PowerShell 建議使用 `curl.exe`，避免呼叫到舊版 PowerShell 的 `curl` alias：

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/metadata
curl.exe -X POST "http://localhost:8000/v1/detect?conf=0.25&iou=0.45" -H "accept: application/json" -F "file=@C:\path\to\image.jpg"
```

Linux 或 macOS：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl -X POST "http://localhost:8000/v1/detect?conf=0.25&iou=0.45" \
  -H "accept: application/json" \
  -F "file=@/path/to/image.jpg"
```

偵測回應格式範例：

```json
{
  "model_name": "mine-yolo-v8",
  "model_version": "1.0.0",
  "image_filename": "image.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "latency_ms": 42.123,
  "warnings": [],
  "detections": [
    {
      "detection_id": 1,
      "class_id": 0,
      "class_name": "m2a4",
      "confidence": 0.912345,
      "bbox_xyxy": {
        "x1": 100.0,
        "y1": 120.0,
        "x2": 300.0,
        "y2": 360.0
      },
      "bbox_normalized": {
        "x1": 0.052083,
        "y1": 0.111111,
        "x2": 0.15625,
        "y2": 0.333333
      },
      "area_px": 48000.0
    }
  ]
}
```

### 回應基本資訊

| 欄位                   | 說明                                                    |
| ---------------------- | ------------------------------------------------------- |
| `model_name`           | 模型名稱，目前為 `mine-yolo-v8`                         |
| `model_version`        | 模型／API 定義版本                                      |
| `image_filename`       | 上傳圖片的原始檔名                                      |
| `image_width`          | 圖片寬度，單位為 pixel                                  |
| `image_height`         | 圖片高度，單位為 pixel                                  |
| `confidence_threshold` | 本次推論使用的信心門檻                                  |
| `iou_threshold`        | 本次 NMS 使用的 IoU 門檻                                |
| `latency_ms`           | 模型 `predict` 執行時間，單位為毫秒，不包含圖片上傳時間 |
| `warnings`             | 非致命問題的警告訊息                                    |
| `detections`           | 偵測到的物件陣列；沒有符合條件的物件時為空陣列          |

### 每筆 detection

| 欄位              | 說明                                                              |
| ----------------- | ----------------------------------------------------------------- |
| `detection_id`    | 本次回應內的流水編號，從 `1` 開始                                 |
| `class_id`        | 模型輸出的類別編號                                                |
| `class_name`      | 根據 `config/class_map.json` 查到的類別名稱                       |
| `confidence`      | 模型對該筆偵測的信心分數，範圍為 `0.0` 至 `1.0`                   |
| `bbox_xyxy`       | Bounding box 的像素座標，包含 `x1`、`y1`、`x2`、`y2`              |
| `bbox_normalized` | 將 bounding box 除以圖片寬高後的正規化座標，範圍為 `0.0` 至 `1.0` |
| `area_px`         | Bounding box 面積，單位為平方像素                                 |

## 驗收測試

先安裝測試相依套件，再執行標準庫 `unittest`：

```powershell
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

測試涵蓋：

- 正常圖片上傳
- 非圖片 Content-Type 回傳 `415`
- 內容損壞的圖片回傳 `400`
- 缺少 `weights/best.pt` 時啟動失敗並顯示明確訊息
- 缺少 `config/class_map.json` 時啟動失敗並顯示明確訊息
- 未定義的 `class_id` 回傳 `class_name: "unknown"` 與 `warnings`
- CUDA 可用時選擇 `cuda:0`，不可用時 fallback 到 `cpu`

測試使用記憶體內圖片與假模型輸出，不會重新訓練模型，也不會下載或修改 YOLO 權重。

## Docker

建立 image：

```powershell
docker build -t mine-detection:latest .
```

以 GPU 啟動 container（主機需安裝 NVIDIA Container Toolkit）：

```powershell
docker run --rm --gpus all --name mine-detection-api -p 8000:8000 mine-detection:latest
```

若主機沒有可用 GPU，可直接啟動；服務會自動 fallback 到 CPU：

```powershell
docker run --rm --name mine-detection-api -p 8000:8000 mine-detection:latest
```

container 啟動後同樣使用 `http://localhost:8000` 呼叫 API。可執行以下命令查看啟動紀錄：

```powershell
docker logs mine-detection-api
```

## 常見問題

### 啟動時顯示找不到 `weights/best.pt`

確認目前位於專案根目錄，並執行：

```powershell
git status
git pull
Test-Path .\weights\best.pt
```

API 不會自動下載模型；`weights/best.pt` 必須實際存在。

### PowerShell 不允許執行 `Activate.ps1`

不需要修改 Execution Policy。直接使用 `.\.venv\Scripts\python.exe` 執行安裝與啟動命令即可。

### API 使用 CPU 而不是 GPU

先呼叫 `GET /metadata` 查看 `device`。只有 PyTorch 能辨識 CUDA 時才會使用 `cuda:0`，否則會自動 fallback 到 `cpu`。

### Port 8000 已被占用

本機執行時可改用其他 port，例如：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

之後請將呼叫網址改為 `http://localhost:8001`。

## 對外部署注意事項

目前 API 沒有 API Key、登入驗證、TLS 與明確的上傳檔案大小限制。若要開放至校外或公網，應先在反向代理或 API Gateway 加入 HTTPS、身分驗證、請求大小限制與流量限制。
