# Open-source tokenizer benchmark

Nine widely used tokenizers measured on the full metric set. The report is
[REPORT.md](REPORT.md); everything needed to regenerate it is here.

| File | What it is |
|---|---|
| `tokenizers.json` | The nine tokenizers, by Hugging Face Hub id |
| `code_ast_config.json` | Maps each programming language to its directory under `code_data/` |
| `fetch_code_corpus.py` | Downloads `code_data/` from `bigcode/the-stack-smol-xs` |
| `run.sh` | The exact command, then the report render |
| `analysis_results.json` | The results the report is rendered from, with `run_metadata` |
| `render_report.py` | Turns the results file into `REPORT.md` |
| `REPORT.md` | Generated. Do not edit by hand |

## Regenerating it

```bash
uv sync --extra flores
hf auth login                                    # FLORES+ and two of the nine
                                                 # tokenizers are gated
uv run python scripts/fetch_flores.py --config configs/core_lang_config.json
uv run python benchmarks/open_source/fetch_code_corpus.py
bash benchmarks/open_source/run.sh
```

`code_data/` is not committed. It holds 1500 files from The Stack, each under
the license of the repository it came from, so it is fetched rather than
redistributed. The FLORES+ corpus under `parallel/` is fetched for the same
reason; see NOTICE.

Two of the nine tokenizers, `meta-llama/Meta-Llama-3-8B` and
`google/gemma-2-9b`, are gated on the Hub and need an accepted license. Drop
them from `tokenizers.json` to run without one: no other number changes, since
every metric here is computed per tokenizer.
