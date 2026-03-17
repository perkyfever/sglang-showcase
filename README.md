# ⚡ SGLang Framework Showcase

## Installation

```bash
pip install --upgrade pip
pip install uv
uv sync --frozen
```

## Running Examples

Launch the server if you are willing to run the code:

```bash
python -m sglang.launch_server \
    --host 127.0.0.1 \
    --port 30000 \
    --model-path your_fav_model # Qwen/Qwen2.5-7B-Instruct
```

All the showcase examples are demonstrated in `demo.ipynb`:

* Inference optimizations benchmarks
    * Continuous batching
    * Radix cache & LPM Scheduling
    * Speculative Decoding

* Structured generation
    ```bash
    uv run python examples/struct_gen.py
    ```

* Generation control flow
    ```bash
    uv run python examples/control_flow.py
    ```

## Sources

* [SGLang Repository](https://github.com/sgl-project/sglang)
* [SGLang Documentation](https://docs.sglang.io/index.html)
* [SGLang Roadmap - 2026 Q1](https://github.com/sgl-project/sglang/issues/12780)
* [Introduction to LLM serving with SGLang - Philip Kiely and Yineng Zhang, Baseten](https://www.youtube.com/watch?v=Ahtaha9fEM0)
