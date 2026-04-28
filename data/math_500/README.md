# MATH-500 test set

自動下載（在專案根目錄執行）：

```bash
python scripts/download_assets.py --math-only
```

或手動放置 `test.jsonl`（每行一個 JSON）。每行應至少包含下列鍵之一：

- `problem`
- `question`
- `instruction`

完整下載資料與模型：`python scripts/download_assets.py`（寫入 `data/math_500/test.jsonl` 與 `models/cache/`）。

Step 0 需要 **至少 50** 筆題目。
