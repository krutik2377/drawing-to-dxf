# Drawing-to-DXF workflow diagrams

Mermaid source for the DXF pipeline, shop-sheet pipeline, and AI layer.  
View in GitHub, VS Code/Cursor (Markdown preview), or [mermaid.live](https://mermaid.live).

---

## A. DXF pipeline (`drawing-to-dxf …` without `sheet`)

```mermaid
flowchart LR
  subgraph ingest [Ingest]
    I[Image or PDF]
  end
  subgraph pre [Preprocess]
    P[Gray, denoise, deskew, resize]
  end
  subgraph geom [Geometry]
    V[Canny + Hough lines]
  end
  subgraph text [Optional OCR]
    O[EasyOCR]
    F[Regex part IDs]
    L[Link segments to parts]
  end
  subgraph out [Output]
    D[ezdxf DXF]
    M[manifest.json]
  end
  I --> P --> V
  P --> O --> F --> L
  V --> L
  V --> D
  L --> D
  L --> M
  V --> M
```

---

## B. Shop sheet pipeline (`drawing-to-dxf sheet …`)

```mermaid
flowchart TB
  subgraph ingest [Ingest]
    S[Image or PDF]
  end
  subgraph pre [Preprocess]
    PR[Gray, denoise, deskew, resize]
  end
  subgraph split [Panel split]
    PS[Contours / gap split]
    PC[Panel crops]
  end
  subgraph perpanel [Per panel]
    O2[EasyOCR excerpt]
    AI{VLM?}
    OLL[Ollama local]
    OAI[OpenAI-compatible API]
    J[JSON fields]
  end
  subgraph compose [Layout]
    R[Pillow grid composite PNG]
  end
  subgraph artifacts [Artifacts]
    PNG[panel_XX.png]
    CMP[composite_layout.png]
    MAN[sheet_manifest.json]
  end
  S --> PR --> PS --> PC
  PC --> O2
  O2 --> AI
  AI -->|ollama| OLL
  AI -->|openai| OAI
  AI -->|none| J
  OLL --> J
  OAI --> J
  PC --> PNG
  J --> R
  PC --> R
  R --> CMP
  J --> MAN
  PNG --> MAN
```

---

## C. Where AI sits (semantic layer — not image generation)

```mermaid
flowchart LR
  CV[OpenCV geometry / split]
  OCR[EasyOCR text]
  VLM[Vision LLM: JSON semantics]
  CAD[DXF or composite PNG]
  CV --> CAD
  OCR --> VLM
  CV --> VLM
  VLM --> CAD
```
