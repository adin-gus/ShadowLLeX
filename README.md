# Shadow LLM

Do you know what composes an LLM? Yeah me neither, that's why I had this thing built with the help of Claude.AI. 97% Vibe Coding, 2% Empirical Science, 1% hot gas — it extracts all the model's layers, embeddings, attention heads, etc. into a clean sorted archive for storage, analysis of model behaviors, or whatever else you can think of to do with it. Part of a larger pipeline still in development which will be announced closer to release.

---

## What It Actually Does

Shadow LLM decomposes a GGUF model into its fundamental information units, classifies each one by functional role, and stores them in a structured library using lossless zstd compression. The library can then be used to reassemble the original model exactly, or (experimentally) compose new models from components across multiple sources.

**The pipeline:**
```
GGUF → Extract → Classify → Store → Reassemble → Load → Run
```

**Validated:** Extracted Mistral 7B, reassembled from library, loaded in llama.cpp, produced coherent output.

**Proof of concept output (chickpea soup prompt):**
```
1. Saute the onion and the garlic in some olive oil in a big pot.
2. Add the chickpeas.
3. Add the vegetable stock and the water.
4. Add the paprika and the cumin.
5. Cook for 20-30 minutes.
6. Add the coconut milk and the lemon juice.
7. Cook for a few more minutes, so the flavors mix.
8. Add salt and pepper.
9. Add some fresh coriander and parsley, cook a few more minutes.
10. Add the lentils, cook for a few more minutes.
11. Done.
```

That's a full model, decomposed to a library, reassembled from scratch, running inference. It works.

---

## The Eight Tiers

Shadow LLM classifies every tensor in the model into one of eight functional tiers, mapped to the axes of a quaternion octahedron (yes, really):

| Tier | What It Is |
|------|------------|
| **CROWN** | Embeddings and output projection — the model's interface with the outside world |
| **INSTRUCTION** | ILA-sensitive layers — how the model interprets and follows direction |
| **REASONING** | Early-mid attention, high kurtosis — where the model thinks |
| **KNOWLEDGE** | Mid MLP layers — factual associations and domain expertise |
| **CONTEXT** | K/V attention — how the model holds context across tokens |
| **STYLE** | Late layers — tone, verbosity, output formatting |
| **FOUNDATION** | Norms, rope, positional encoding — structural skeleton |
| **VOID** | Holographic-quiet — distributed low-signal representations |

Classification is heuristic — based on layer depth, ILA scores, scale envelope sampling, and function name matching. It's a starting point, not ground truth. See DeepSeek's critique in the docs folder for known limitations and improvement directions.

---

## Library Structure

```
shadow-library/
  {model-name}/
    CROWN/
      tensors/          ← zstd-compressed tensor files (.bin.zst)
      CROWN_records.json
    KNOWLEDGE/
      tensors/
      KNOWLEDGE_records.json
    ... (all 8 tiers)
    blueprint.json      ← full model metadata for reassembly
  global_index.json     ← cross-model queryable index
```

Storage is lossless. Every tensor is stored exactly as extracted, compressed with zstd. Blueprint contains all GGUF metadata needed for reassembly without the original model file.

---

## Install

```bash
pip install zstandard gguf tqdm
```

Optional (for semantic search in inspector):
```bash
pip install memvid
```

---

## Usage

**Extract a model:**
```bash
python alloy_shadow_extract.py --model path/to/model.gguf --out-dir shadow-library
```

**Inspect the library:**
```bash
python shadow_inspector.py --library shadow-library
```

**Reassemble a model:**
```bash
python alloy_shadow_compose.py --library shadow-library --blueprint model-name --out rebuilt.gguf
```

**Interactive toolbox (Windows):**
```
shadow_toolbox.bat
```

---

## What It Doesn't Do Yet

- Cross-model composition is experimental — shape mismatches between different architectures require projection layers not yet fully implemented
- Tier classification accuracy is unvalidated against activation probing — the heuristics are reasonable but not proven
- No GPU acceleration during extraction — pure CPU, ~70 seconds for a 7B model
- The "smart validator" (automatic metadata repair for assembled models) is planned but not built

---

## Attribution

**Concept and direction:** Adin Gus ([@adin-gus](https://github.com/adin-gus))

**Architecture, implementation, and debugging:** Claude Sonnet 4.6 (Anthropic) — written interactively over several sessions. The theoretical framework (quaternion tier mapping, Diamond Haystack, Shadow LLM decomposition concept) emerged from collaborative development. Claude wrote the majority of the code; Adin Gus specified the problems, validated the outputs, and made the architectural decisions.

This is what "vibe coding" looks like when you push it far enough — a novel approach to LLM decomposition that neither party would have arrived at alone.

**Additional analysis:** DeepSeek R1 — provided critique of the classification system and suggested improvements (documented in the tier classification comments).

---

## Part of a Larger Pipeline

Shadow LLM is the storage and decomposition layer of a larger system under development. More details closer to release.

---

## License

MIT. Do whatever you want with it. If you improve the tier classification or crack cross-architecture composition, please PR — that's exactly the kind of help this project needs.
