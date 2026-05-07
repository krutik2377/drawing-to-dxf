# drawing-to-dxf

**Engineering drawing (image or PDF) → Gemini vision → Python script per component → DXF files + optional assembly DXF.**

No AutoCAD required to generate DXF (open in AutoCAD, BricsCAD, LibreCAD, etc.).

## Install

```powershell
cd C:\Users\admin\Desktop\HyzenInterview
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Set **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`**.

## Usage

```powershell
drawing-to-dxf "path\to\sheet.png" -o out_gemini_codegen
```

or

```powershell
drawing-to-dxf gemini-codegen "path\to\sheet.png" -o out_gemini_codegen
```

**PDF:** `drawing-to-dxf draft.pdf --pdf-page 0 --pdf-dpi 200 -o out`

**Useful flags:** `--gemini-model`, `--max-side`, `--gemini-max-output-tokens`, `--layout-gap-mm`, `--gemini-timeout`, `--script-timeout`

See **`docs/CLEAN_AND_RUN.md`** for clean output cycles and a results table.

## Security

The tool **executes model-generated Python** on your machine. Use trusted inputs/models or an isolated environment.

## Tests

```powershell
pip install -e ".[dev]"
pytest -q
```

## License

Use and modify for interview or internal work; add a license if you redistribute.
