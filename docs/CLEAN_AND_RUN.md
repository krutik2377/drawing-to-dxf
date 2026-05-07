# Clean and run — `drawing-to-dxf`

Reset outputs and run the **Gemini → Python → DXF** flow. Repo root: `HyzenInterview` (adjust paths if needed).

## Prerequisites

- Python **3.10+**
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment

## Clean outputs

Removes **`out_gemini_codegen/`**, legacy **`out/`**, **`out_sheet/`**, **`out_panels/`**, and **`.pytest_cache/`**:

```powershell
python scripts/clean_outputs.py
```

Full venv reset: `python scripts/clean_outputs.py --include-venv` then recreate the venv.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Developers (tests): `pip install -e ".[dev]"`

## Run

```powershell
$env:GEMINI_API_KEY = "YOUR_KEY"
drawing-to-dxf "C:\path\to\drawing.png" -o out_gemini_codegen
```

Same with explicit subcommand:

```powershell
drawing-to-dxf gemini-codegen "C:\path\to\drawing.png" -o out_gemini_codegen
```

**PDF:** add `--pdf-page 0 --pdf-dpi 200`

**Large responses:** `--gemini-max-output-tokens 32768`  
**Sharper vision input:** `--max-side 3072` (or `0` for full resolution)

## Outputs (under `-o`)

| Path | Meaning |
|------|--------|
| `*_gemini_codegen_manifest.json` | Run metadata |
| `*_gemini_codegen_raw.txt` | Raw model response |
| `gemini_codegen_scripts/` | Generated `.py` per part |
| `gemini_codegen_dxfs/` | Per-part `.dxf` |
| `*_gemini_codegen_assembly.dxf` | Merged layout (when merge succeeds) |

## Tests

```powershell
pip install -e ".[dev]"
pytest -q
```
