# Repository Guidelines

## Project Structure & Module Organization
本仓库是一个 Python 学术配图助手，核心代码按职责拆分：
- `agents/`：多 Agent 流水线实现（如 `planner_agent.py`、`critic_agent.py`）。
- `providers/`：模型服务提供商抽象与实现（`base.py`、`evolink.py`）。
- `utils/`：配置、处理流程与通用工具（如 `config.py`、`paperviz_processor.py`）。
- `prompts/` 与 `style_guides/`：提示词与学术风格规则。
- `tests/`：单元测试（当前以 `test_evolink_provider.py` 为主）。
- `scripts/`：启动脚本；`demo.py` 为 Streamlit UI，`main.py` 为批处理入口。
- 运行产物默认输出到 `results/<dataset>_<task>/`。

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`：创建并激活虚拟环境。
- `pip install -r requirements.txt`：安装依赖。
- `streamlit run demo.py --server.port 8501`：启动本地 Web 演示。
- `bash scripts/run_demo.sh`：使用 `uv` 自动补齐环境并启动演示。
- `bash scripts/run_main.sh`：运行批处理实验（`main.py` 默认参数）。
- `pytest tests/test_evolink_provider.py -v`：运行 Provider 核心测试。

## Coding Style & Naming Conventions
- Python 使用 4 空格缩进，遵循 PEP 8；保持与现有代码一致。
- 文件名使用 `snake_case.py`，类名使用 `PascalCase`，函数与变量使用 `snake_case`。
- 异步逻辑统一使用 `async/await`，避免在协程中混入阻塞式 I/O。
- 新增 Provider/Agent 时优先复用抽象接口：`BaseProvider`、`BaseAgent`。

## Testing Guidelines
- 测试框架为 `pytest`（含 `pytest.mark.asyncio` 异步用例）。
- 测试文件命名 `test_*.py`，测试函数命名 `test_*`。
- 新增能力至少覆盖：成功路径、重试/失败路径、关键边界输入。
- 涉及外部 API 的逻辑应使用 mock，避免在 CI/本地测试中真实调用。

## Commit & Pull Request Guidelines
- 提交信息遵循现有历史风格：`feat:`、`fix:` 前缀，后接简短中文描述。
- 单个 commit 聚焦一件事，避免把重构与功能改动混在一起。
- PR 需包含：变更目的、核心改动、测试命令与结果。
- 若修改 UI（`demo.py`/`static/`），请附截图或录屏；若关联问题请链接 Issue。
- 按 `CONTRIBUTING.md` 要求，提交前确认 CLA 与代码评审流程。

## Architecture Notes
- 典型执行链路为 `Retriever -> Planner -> Stylist -> Visualizer -> Critic`，可在 `demo.py` 中看到并行候选生成流程。
- 实验参数由 `utils/config.py` 的 `ExpConfig` 统一管理，避免在 Agent 内硬编码模型名或路径。
- 新增功能时优先在 `utils/` 放通用逻辑，在 `agents/` 仅保留编排与策略，减少耦合。
- 命名建议示例：分支名 `feat/evolink-retry`、`fix/streamlit-timeout`，文件名如 `test_provider_retry.py`。

## Security & Configuration Tips
- 不要提交真实 API Key；本地配置使用 `configs/model_config.yaml`（可由 `model_config.template.yaml` 复制）。
- 提交前检查 `.gitignore` 与配置文件 diff，避免泄露密钥和本地路径。
