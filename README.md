# qharness · A 股量化研究 Harness

研报 → 因子假设 → 写因子代码 → 回测验证 → 反思迭代 → 登记因子 → 每日目标持仓。重点不是让代码跑通，而是**让结论可信**：评价和生成隔离、口径用硬规则查、多重检验有校正、经验跨轮积累。开发计划见 [PLAN.md](PLAN.md)。

全部使用真实模型和真实行情，没有 mock。

## 快速开始

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt
```

首次准备数据（沪深 300，2023-01 起，约 1 小时，可断点续传）：

```bash
.venv/bin/python -m qharness.market --start 2023-01-01 --end 2026-10-29
.venv/bin/python -m qharness.market --source akshare
.venv/bin/python -m qharness.market --source cap --start 2023-01-01 --end 2026-10-29
.venv/bin/python -m qharness.market --source industry
```

依次是：日线（baostock，被限流时第二条用新浪补齐）、不复权价与流通股本（算市值、判涨跌停）、申万一级行业历史。

研究：

```bash
.venv/bin/python -m qharness demo --env-file path/to/.env
```

```bash
.venv/bin/python -m qharness run --report 研报.md --env-file path/to/.env
```

每日（收盘后，可放进 cron）：

```bash
.venv/bin/python -m qharness update && .venv/bin/python -m qharness daily --paper
```

## 命令

| 命令 | 作用 | 产出 |
|---|---|---|
| `run --report 研报.md` / `--hypothesis h.json` | 研报 → 假设 → 实验循环 → 策略对照 → 通过且测试段成立的因子登记 | `runs/<run_id>/`：`summary.md`、`trace.jsonl`、`results.json`、`strategy.json`、`llm_calls.json` |
| `demo` | 用 `samples/` 里的示例研报跑 `run` | 同上 |
| `update` | 刷新最新沪深 300 成分；新成分股补全历史；已有股票追加最近交易日；有 Qlib 目录时一并重新生成 | `data/hs300/`、`data/qlib_cn/` |
| `qlib-dump` | 行情缓存转成 Qlib 数据目录（之后 `run` / `daily` 自动启用 Qlib，`--no-qlib` 关闭） | `data/qlib_cn/` |
| `qlib-train` | Qlib 模型训练：Alpha158 特征 + 梯度提升树，沿用 Harness 的切分和测试段纪律，和 5 个基础因子的等权基线比较（`--with-registry` 把 Qlib 表达式写的登记因子加进特征） | `runs/qlib_model/<时间>/`：`summary.json`、`pred.pkl`、`mlruns/` |
| `daily` | 因子库 + 登记因子等权合成，在最新成分里选 TopK，对比上次持仓给出买卖清单 | `portfolio/<日期>.csv` |
| `daily --paper` | 同上，并更新模拟账户：上次挂单在之后第一个交易日开盘撮合、逐日记净值，到调仓周期挂新单 | `portfolio/account.json`、`account.trades.csv` |
| `paper-sim` | 模拟账户历史回放（默认测试段），输出净值、成交记录，并和同区间回测对照 | `runs/paper_sim/` |
| `factors` | 查看登记因子（代码、各段指标、来源 run） | — |
| `eval-research` | 研究层评测集打分 | `runs/research_eval.json` |
| `experience` | 查看经验库 | — |

- **模型**：默认 Anthropic Messages 兼容接口（读 `ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / ANTHROPIC_MODEL`，`--env-file` 或环境变量 `QH_ENV_FILE` 指定 .env）；`--llm openai` 走 OpenAI 兼容接口（读 `QH_LLM_*`）。默认关闭思考，`QH_LLM_THINKING=default` 恢复。每次调用的耗时和 token 记在 `llm_calls.json`；示例研报一次完整运行约 16 次调用、6~7k token。
- **研报**：PDF（`report_pdf.py`）、PPT（`.pptx`，`report_ppt.py`）都会自动转成带页码和章节的文本，转换结果存到 `runs/<run_id>/report.md` 留档；也可以直接给 Markdown/TXT（用 `<!-- page: N -->` 标页码）。`samples/` 里的示例研报是虚构文本（`report_volume_price.pptx` 是同一篇用 Keynote 做成的 PPT）。
- **PPT 解析**：`.pptx` 是 zip 包里的 XML，用标准库直接解析，不依赖 python-pptx。
  - 页序按放映顺序；每页的标题占位符当章节标题，其余文本框按版面从上到下排列，表格转成【表格】，图表和图片只计数。
  - 演讲者备注并到该页末尾，标成「演讲者备注（与本页正文同等对待）」。只写「备注：」时，模型两次都把这段跳过了。
  - 标题里没有「核心观点 / 摘要 / 投资要点」时，第一张有正文的页标成摘要，供主题定位用。
  - 实测（Keynote 导出的 pptx）：抽出 2 个观点，第一个观点的出处合并了第 2 页正文和第 3 页的演讲者备注；两个都写成 Qlib 表达式，被研究层淘汰（量价相关太弱；5 日动量正好是 5 日反转取负）。

## 数据

| 内容 | 来源 | 说明 |
|---|---|---|
| 股票池 | baostock `query_hs300_stocks(起始日)` | 研究和回测只用起始日成分（含之后调出、退市的），避免幸存者偏差；`daily` 用中证指数官网的最新成分 |
| 日线（后复权） | baostock 为主，akshare/新浪补齐和每日更新 | 两源抽样对比：日收益最大差 <0.1%，成交量一致（`market.verify_sources`）；增量更新按重叠日收盘价比值拼接 |
| 流通市值、换手率 | 新浪不复权日线的收盘价 × 流通股本 | 换手率作为可选字段 `turnover` 开放给因子代码 |
| 行业 | 巨潮资讯行业变更历史，申万一级；无申万记录时用新财富分类 | 按变更日期还原每天的行业 |
| 涨跌停 | 不复权价与前收计算（主板 10%，创业板/科创板 20%） | ST 的 5% 未处理 |

baostock 并发太高会封 IP，东财接口会断连接，所以下载都是串行加间隔。缓存里每个文件的 `source` 列标明来源。

## 流程

```
模块一 研报 SOP（固定节点）
  PDF 转文本(pdfplumber; 目录识别章节; 图表块里数字行为主的转成【表格】、其余的图去掉;
            数学斜体字母归一、拆行的分数收拢成带名字的【公式】; 去页眉/联系方式/免责声明) → 清洗切片(带页码)
  → 主题定位 → 观点提取(论据须能在原文找到) → 观点归并 → 逻辑树
  → 逐个观点判断能否写成个股截面因子(附上研报里的公式定义和最相关的 2 段原文)
       字段不可得、择时类、依赖日内/逐笔数据的 → 不可检验, 写明缺什么(日内依赖由程序按关键词兜底)
  → 忠实性核查(研报没直接说的变量、只能近似的变量剔除) → 去重
模块二 实验编排（两层循环）
  ExperimentAgent: 通过原生 tool calling 调用 run_factor 提交代码 → 沙箱子进程执行(静态检查/行序不变/未来函数截断)
                   → 执行结果以 tool_result 回传 → 出错修复 ≤3 次                                  ← 内环：修代码
  按因子值去重: 与本次已评估因子截面相关 ≥0.98 视为同一因子跳过；经验库只跳过公式/方向/字段/窗口相同的已淘汰因子
  评估: 市值+行业中性化 → RankIC / Newey-West t / 分年 IC / 分层 / 换手 / 与 5 个已有因子的相关性
  ResearchAgent:   程序核对的关键事实 + 结构化指标 → 预注册规则判定 → accept / refine / reject   ← 外环：修假设 ≤3 轮
                   验证段 t 门槛随试验次数提高；修改必须对应一条诊断；两段都为负才允许翻转方向
  accept 后测试段开放一次；测试段成立才登记。成功实现、修复案例、淘汰方向、纠正记录写入经验库
模块三 策略
  基线(因子库 + 已登记因子 TopK 多头) vs 基线 + 新因子，测试段比较；成交日停牌/涨停买不进、停牌/跌停卖不出
  daily: 最新收盘后的目标持仓与买卖清单
```

## 编排（LangGraph）

研报 SOP 和实验编排都是 LangGraph `StateGraph`，节点即 trace 里的 `sop.*` / `experiment.*` / `research.*`。图结构可以直接打印：

```bash
.venv/bin/python -c "from qharness.report_sop import ReportSOP; print(ReportSOP(None, None).graph.get_graph().draw_mermaid())"
```

```
研报 SOP（report_sop.ReportSOP）
  chunk → topic → viewpoints → merge ─(没有观点)→ finalize
                                     └→ logic_tree → hypothesis → faithful → finalize

实验编排（orchestrator.FactorHarness，每个候选一次 invoke）
  check_rejected ─(同一因子已淘汰)→ END
        └→ apply_correction → generate → execute ─(失败, 未超限)→ fix → execute     内环：修代码 ≤3
                                             ├(失败, 超限)→ give_up
                                             └(成功)→ dedup ─(与已评估因子重复)→ END
                                                        └→ judge ─ accept / reject → END
                                                                 └ refine → generate / step_limit   外环：修假设 ≤3
```

- **上下文隔离**：两个 Agent 各自持有对话历史，图的状态里只放层间传递的结构化字段（修改后的假设、过滤过指标的修改说明、结构化指标）。
- **上限**：修复次数、调整轮数、总步数都在条件边的路由函数里；`recursion_limit` 按总步数放宽，真正的兜底是 `max_steps`。

### Qlib

`qlib_bridge.py`，pyqlib 0.9.7。`qlib-dump` 之后自动启用：

- **数据**：行情缓存写成 Qlib 的二进制目录（日历、股票列表、每只股票每个字段一个 `.day.bin`）。价格是后复权价，`$factor` = 后复权 / 不复权；另存 `$limit_up` / `$limit_down`（当天涨跌停或停牌为 1）给回测用。`update` 之后会自动重做。
- **因子计算（第二个工具）**：实验层多一个 `run_qlib_factor(expression)` 工具，因子可以写成 Qlib 表达式（如 `-Corr($close,$volume,10)`），由 Qlib 计算。提示词要求能用表达式写清楚的优先用表达式。
  - 表达式先过白名单：只允许 Qlib 的时序算子和行情字段；`Ref` 偏移为负直接判为未来函数；Qlib 不支持的一元负号自动改写成 `-1*`。
  - 算完照常做覆盖率、截面常数和截断重算检查，和 Python 代码走同一套沙箱。
  - Qlib 读的是自己的数据目录：每次计算顺带取 `$close` 和当前行情比对，对不上（比如更新后没重新生成）就报错，不会静默算出另一份数。
  - 核对：5 日反转、20 日均量、20 日波动三个因子，Qlib 和 pandas 写法相关系数都是 1.0，差异在 float32 精度内。Qlib 的窗口算子在窗口不足时也出值，覆盖率略高。
- **回测复核**：`strategy.json` 里多一份 `qlib`，用 Qlib 的 `TopkDropoutStrategy` + `SimulatorExecutor` 对同一组信号回测：次日开盘成交、100 股整手、买 0.125% / 卖 0.175% 费用、涨跌停和停牌不成交，基准是股票池等权。测试段 5 个基础因子的结果分解：

  | | 自写回测 | Qlib 回测 |
  |---|---|---|
  | 收盘价成交、零费用 | +5.2% | +3.4% |
  | 收盘价成交、有费用 | -2.3% | -3.8% |
  | 次日开盘成交、有费用 | — | -6.5% |

  费用两边都扣掉约 7 个百分点。同口径下剩下约 1.8 个百分点的差距来自持仓方式：自写回测每天按等权计算收益，相当于每天免费再平衡；Qlib 是整手买入后随价格漂移，并留约 1% 现金。按开盘价成交又少约 2.7 个百分点。
  - 接入时踩过的坑：把非调仓日的信号置空，Qlib 会在调仓日的下一天再换一次仓（它取「上一根 bar 里最后一个信号」），年化超额一度算成 -21.9%。现在调仓日的分数前向填充到下一个调仓日。
- **实测**（真实模型 + 真实行情）：两个候选并行，模型都选了 `run_qlib_factor`，分别提交 `-Corr($close,$volume,10)` 和 `Skew($close/Ref($close,1)-1,20)`。后者的假设方向是 -1，模型漏了负号。之后把方向改成由 Harness 施加（见「因子方向由 Harness 施加」）。

### Qlib 模型训练

`qlib_model.py`，`qharness qlib-train`。对应 Qlib `qrun` 的工作流：数据处理器 → 数据集 → 模型 → `SignalRecord`（存预测）→ `SigAnaRecord`（算 IC），记录在 MLflow 里。

- **特征**：Alpha158（158 个量价特征）。它要用成交均价，所以 Qlib 数据目录多了 `$vwap` = 成交额 / 成交量 × 复权因子（核对过：成交额 / 成交量和不复权收盘价之比中位数 1.000，成交量单位是股）。`--with-registry` 再加上因子库里 Qlib 表达式写的登记因子（登记代码已含方向，不再处理）。
- **标签和切分**：和 Harness 同一个标签 `Ref($close,-6)/Ref($close,-1)-1`，同一套训练 / 验证 / 测试段，每段末尾去掉 6 天。
- **模型**：LightGBM 能加载就用 Qlib 的 `LGBModel`。这台机器上 LightGBM 加载失败（macOS 缺 OpenMP 运行库 libomp），自动改用 sklearn 的 `HistGradientBoostingRegressor`。两者是同一类直方图梯度提升树，同样按验证段 L2 早停（每 20 棵树看一次，连续 3 次不降就停，用最优轮数重训）。装上 libomp（`brew install libomp`）后 `--model auto` 会切到 LightGBM。
- **评估**：预测值当作一个因子交给 `Evaluator`：中性化 RankIC、Newey-West t、和基础因子库的相关性。测试段同样只开一次，锁的键包含模型、种子、额外特征、切分和超参，换配置会留下新的记录（`runs/qlib_model/test_opened.txt`）。
- **回测**：自有回测和 Qlib 回测各一份，和基线比较。另外报「调仓相位」：5 天调一次仓，从测试段第 0~4 天起各回测一遍，给出区间。测试段不到一年，起点不同能差十几个点，单个数字不可靠。

实测（测试段 2026-01-30 ~ 2026-10-22，179 个交易日，379 只股票，训练约 30 秒）：

| | 测试段 RankIC（中性化） | Newey-West t | 年化超额（自有回测） | 5 个调仓起点 | Qlib 回测（次日开盘、整手、费用） |
|---|---|---|---|---|---|
| 5 个基础因子等权（基线） | 0.030 | 1.4 | -2.3% | -4.1% ~ +2.1% | -6.5% |
| Alpha158 + 梯度提升树 | 0.042 | 3.5 | +29.6% | +13.9% ~ +40.3% | +28.7% |

- 未中性化的 RankIC 是 0.086（Qlib 的 `SigAnaRecord` 给出同样的 0.086），中性化后减半，一半的预测力来自市值和行业暴露。和 5 个基础因子的最大相关 0.32。
- 验证段 L2 从第一次检查（20 棵树）就开始上升，模型只保留了 20 棵树，很浅。
- 超额很高，所以专门查了前视（脚本不在仓库里，结论如下）：
  - 截断重算：测试段两个日期，只用到当天的数据重新算 158 个特征，和全量计算完全一致（最大差 0）。
  - 信号延迟：用 1 / 3 / 5 / 10 天前的预测交易，年化超额 +53% / +42% / +25% / +11%，逐步衰减，没有断崖。
  - 分月：9 个月里 6 个月为正，7 月 -7.9%。
- 结论：排序能力在测试段统计上成立（t 3.5），但年化超额的点估计来自不到一年的样本，调仓起点不同就在 +14% ~ +40% 之间，不能当作可预期收益。
- 测试段纪律的说明：开发时同一配置在临时目录里跑过两次（第一次跑通、第二次加相位统计），中间没有改模型和特征；正式记录是 `runs/qlib_model/` 下的这次。

### 因子方向由 Harness 施加

实验层看不到 `direction`，只按 `calculation` 原样实现；执行前 `sandbox.with_direction` 按假设方向包一层（Python 代码包一个取负的 `factor`，Qlib 表达式包成 `-1*(…)`），之后的评估、去重、测试段、登记都用包过的版本。

- **为什么**：模型自己处理方向会漏负号，而且这种错规则查不出来。
- **只翻方向不重写代码**：研究层的修改只改了方向时，图从 `refine` 直接回到 `execute`，原代码不变、不调模型（trace 里记 `research.flip_only`）。
- **经验库口径**：成功实现存的是不含方向的原始代码；以前由模型自己处理方向的成功实现不再被检索，避免照抄后方向算两遍。
- **实测**：同一个偏度因子写成方向 -1 和方向 +1 两个假设。模型两次写的都是同一个 `Skew(...)`，Harness 施加方向后两者截面相关 -1.000，被按值去重。方向写反的那个，研究层只翻了方向，直接重新执行，没有再调写代码的模型。

### 实验层的 Tool Calling

- **怎么提交**：写代码的一层通过 Anthropic Messages 的原生 tool calling 调用唯一的工具 `run_factor(code)`。
- **工具返回什么**：只有执行状态、报错和口径检查（覆盖率、行序不变、未来函数截断）。回测指标由 Harness 在工具一侧计算，直接交给研究层，写代码的一层看不到评价标准。
- **循环由谁控制**：仍由编排图控制，不让模型在一次调用里自己无限重试：
  1. 每次 `tool_use` 对应图里的一次 `execute` 节点；
  2. 执行结果作为 `tool_result` 放在下一条 user 消息最前面，和检索到的修复经验一起回传；
  3. 修复上限、经验检索、检查点都照常生效。
- **兼容**：不支持工具的接口（`--llm openai`）自动退回文本模式，从代码块里取代码。trace 里的 `experiment.submit` 节点会记下每次提交走的是 `tool_call` 还是 `text`。
- **实测**：故意给一个会诱导写出居中窗口的假设：
  1. 第一次提交被判为 LOOKAHEAD，报错以 `tool_result` 回传；
  2. 模型再次调用 `run_factor` 提交修正后的代码，执行成功；
  3. 这次修复作为 LOOKAHEAD 类经验存进了经验库。

### 模拟交易账户

`paper.py`：T 日收盘后挂单，T+1 日开盘撮合的多头账户。

- **价格**：用不复权价成交和记市值（后复权开盘价 × 当天复权系数，取整到分），按 100 股整手。
- **成交限制**：停牌不成交；开盘即涨停买不进，开盘即跌停卖不出。
- **费用**：佣金万 2.5（最低 5 元）、卖出印花税 0.05%、单边滑点 0.1%。
- **分红送转**：复权系数变化时按比例调整股数，相当于分红再投资。
- **落盘**：现金、持仓、挂单、每日净值写在 `account.json`，成交记录追加到 CSV，每天接着上次的状态继续。
- **核对**：
  - **记账**：183 个交易日（其中 35 天持仓遇到分红送转），账户净值日收益和按市值加权的后复权持仓收益完全一致。
  - **和回测对照**（2026-01-21 ~ 2026-10-21，5 个基础因子，TopK 20，每 5 天调仓）：模拟账户年化超额 +0.69%、超额最大回撤 -10.5%；同区间回测 +0.47%、-11.2%。
  - **真实交易日**：账户 2026-10-29 收盘挂单，10-30 开盘 20 笔全部成交（成交额 98.0 万，费用 245 元），10-30 收盘净值 997,856。之后每天收盘后 `update && daily --paper` 接着跑。注意新浪一般收盘后 1~2 小时才发布当天数据，太早运行 `update` 会显示 `no_new_data`。
  - **为什么之前对不上**：不加滑点时，模拟账户的年化超额是 +5.5%。差距主要来自回测按单边 0.15% 扣成本，而账户只算了佣金和印花税；加上滑点后两者对齐。

### 候选并行

`run --workers N`（默认 1）：候选之间用线程池并行，每个候选还是一张独立的图、一个检查点 thread。时间主要花在等模型返回，因子执行在子进程里，所以线程就够用。

- **共享状态加锁**：经验库读写、trace 编号、模型调用记录、测试段锁、按值去重这几处加了锁。两个相同因子同时到去重节点时，只放过先到的那个。
- **trace**：并行时每条事件带 `cand` 字段（`cand-0`、`cand-1`…），能按候选拆开看。
- **代价**：同一批候选看不到彼此刚沉淀的经验；去重按谁先到算，所以被判为重复的是哪一个不固定。结果仍按输入顺序返回。
- **实测**：真实行情、4 个候选（其中两个是措辞不同的同一个 20 日动量）：串行 48 秒，4 路并行 15 秒，都是 7 次模型调用；两次的判定完全一致，重复的那个两次都被按值去重跳过。
- **沙箱环境故障单独报**：沙箱改成单个子进程 + 管道。子进程启动就崩（比如调用方脚本缺 `if __name__ == '__main__'`）时立刻报 `SANDBOX`，不再进修复循环。以前用进程池会等到超时、误报成 `TIMEOUT`，模型就去「优化速度」，四个候选白白各修了 3 次。

### 检查点与续跑

每次 `run` / `demo` 都把检查点写进 `runs/<run_id>/checkpoints.sqlite`（SQLite，每个节点执行完同步落盘）。研报 SOP 一个 thread，每个候选因子一个 thread（序号 + 假设内容哈希）。进程被打断（报错、Ctrl-C、`kill -9`）后：

```bash
.venv/bin/python -m qharness run --resume <run_id> --env-file path/to/.env
```

- **沿用原参数**：研报、数据、经验库等从 `args.json` 读取，研报文本用留档的 `report.md`。
- **哪些会跳过**：已完成的研报阶段和候选直接取结果，跑到一半的候选从断点节点继续。
- **续跑时保持正确的几处**：
  - 测试段锁落盘到 `test_opened.txt`，换进程也只能开一次；
  - 本次运行的去重列表从已完成候选重建；
  - 模型调用每次当场追加到 `llm_calls.jsonl`，进程被杀也不丢。
- **状态可序列化**：只放普通类型。两个 Agent 只存对话历史，节点里重建；因子值不进状态，需要时按代码重算。
- **人工审核**：`FactorHarness(..., interrupt_before=["judge"])` 让每个候选在研究层判定前停下，看过回测结果后用同一个检查点续跑。
- **不写检查点**：加 `--no-checkpoint`。

实测：用中银研报跑到第 4 个候选时 `kill -9`，续跑恢复了研报阶段和 3 个已完成候选，第 4 个从「生成代码」继续，后面 4 个按值去重时对上了恢复回来的因子。用示例研报在代码执行完时 `kill -9`，续跑直接从研究层判定开始，被杀前的 14 次调用都保留了。

## 设计点 ↔ 代码 ↔ 测试

| 设计 | 落在哪 | 怎么验证 |
|---|---|---|
| 未来函数硬规则 | `sandbox.py`：静态拒绝负 shift / 居中窗口 / 引用 label；截断两次重算，历史值必须不变 | `test_sandbox_rejects_common_mistakes`、`test_bfill_lookahead_is_caught_dynamically` |
| 没按股票分组 | `sandbox.py`：换行序重算，结果必须不变 | 同上 |
| 执行隔离 | `sandbox.py`：spawn 子进程 + CPU/内存上限 + 整体超时杀进程 | `test_sandbox_kills_runaway_code` |
| 中性化 | `risk.neutralize`：逐日对 [市值, 行业哑变量] 回归取残差；同时报未中性化 RankIC | `test_neutralize_removes_size_and_industry`、`test_neutralized_evaluation_reports_raw_ic` |
| 重叠标签的显著性 | `evaluate.newey_west_t`（滞后 = 标签期 5 天） | `test_newey_west_accounts_for_overlap` |
| 验证段多重检验 | `Thresholds.t_valid(n)`：Bonferroni，n 为该假设已试验次数（含历史纠正前的试验） | `test_research_rules` |
| 修改建议的约束 | `ResearchAgent.check_revision`：无对应诊断、空修改、两段方向不一致时翻转，都由规则驳回 | `test_revision_guard` |
| 规则优先 | `ResearchAgent.rules`：阈值预注册，模型不能推翻 | `test_research_rules_override_real_model` |
| 研究层判断质量 | `research_eval.py`：重复因子、噪声、市值代理、方向写反四类已知答案的用例 | `qharness eval-research` |
| 上下文隔离 | 两层各自历史；写代码的一层看不到指标和阈值；修改说明过滤指标句子 | `test_end_to_end_real_model` |
| Co-STEER 经验检索 | `experience.py`：成功实现 / 修复案例 / 淘汰方向 / 纠正记录 | `test_end_to_end_real_model`、`test_correction_matching` |
| 测试段只开一次 | `Evaluator.evaluate_test` 按因子加锁（可落盘）；每段末尾留 embargo | `test_good_factor_runs_and_test_set_opens_once`、`test_test_set_lock_survives_restart` |
| 检查点续跑 | `checkpoint.py`（SQLite）、`--resume` | `test_checkpoint_resume_from_new_instance`、`test_candidate_resumes_at_judge_real_model` |
| 可交易性 | `strategy.pick`：卖不出的留下，买不进的顺延 | `test_pick_respects_tradability`、`test_backtest_with_tradability` |
| 因子登记 | `registry.admit`：测试段 RankIC > 0 且 t ≥ 0 | `test_registry_admission` |
| Tool Calling | `agents.ExperimentAgent`（`run_factor` 工具，tool_result 不含指标） | `test_experiment_agent_native_tool_calling` |
| 模拟交易 | `paper.py`：次日开盘撮合、整手、费用滑点、停牌涨跌停、分红送转、状态落盘 | `test_paper_fills_next_open_with_lots_and_fees`、`test_paper_blocks_halted_and_limit_up`、`test_paper_corporate_action_keeps_value` |
| 研报观点忠实 | `report_sop`：论据须出自原文；假设逐个对照观点核查；用到不可得字段、依赖日内/逐笔数据的由程序挪到不可检验 | `test_chunking_and_grounding`、`test_needs_intraday_guard`、`test_report_sop_real_model` |
| PDF 公式与表格 | `report_pdf.pages_to_text` | `test_pdf_formula_and_table_parsing` |
| PPT 输入 | `report_ppt.pptx_to_text`：页序、标题、表格、演讲者备注、摘要定位 | `test_pptx_parsing_keynote_export`、`test_pptx_table_and_summary_fallback` |
| 方向由 Harness 施加 | `sandbox.with_direction`；实验层提示词不含 direction；只翻方向时跳过重新生成 | `test_direction_applied_by_harness`、`test_coder_does_not_see_direction` |
| Qlib | `qlib_bridge.py`：数据转换、表达式白名单、和行情一致性检查、TopkDropout 回测复核 | `test_qlib_expression_matches_pandas`、`test_qlib_expression_static_rules`、`test_qlib_rejects_stale_data`、`test_qlib_backtest_runs`、`test_experiment_agent_uses_qlib_tool_real_model` |
| Qlib 模型训练 | `qlib_model.py`：Alpha158 + 梯度提升树（LightGBM 不可用时 sklearn），Qlib workflow 记录，Harness 口径评估，测试段按配置加锁，调仓相位区间 | `test_qlib_model_train_alpha158`、`test_registry_exprs_keep_registered_direction` |
| 候选并行 | `FactorHarness.run(workers=N)`：线程池；共享状态加锁；trace 带候选标签 | `test_duplicate_check_is_atomic_across_threads`、`test_tracer_tags_events_per_thread`、`test_parallel_candidates_real_model` |
| 去重 | `orchestrator.factor_key`（同一因子才跳过已淘汰方向）、`_duplicate_of`（按因子值） | `test_factor_key_same_factor_only`、`test_duplicate_by_values` |

## 测试

```bash
.venv/bin/python -m pytest -q
```

```bash
QH_ENV_FILE=path/to/.env .venv/bin/python -m pytest -q
```

第一条只跑确定性部分，在 `tests/fixtures` 的真实行情切片上跑，无需网络；切片用 `tests/fixtures/make_fixture.py` 从缓存生成。第二条加上调真实模型的测试，只断言流程约束，不断言模型的具体判断。研究层判断质量用 `eval-research` 单独打分。

## 研究层评测集结果（deepseek-v4-pro，关闭思考）

| 用例 | 预期 | 加「关键事实」前 | 加之后（3 次） |
|---|---|---|---|
| 重复因子：5 日反转 | 高相关淘汰，诊断指出重复 | 1.0 | 1.0 / 1.0 / 1.0 |
| 噪声因子 | 不接受，不靠翻方向修 | 1.0 | 1.0 / 1.0 / 1.0 |
| 市值代理（成交额规模） | 诊断指出市值/流动性暴露 | 0.5 | 1.0 / 1.0 / 1.0 |
| 方向写反（收益偏度） | 建议翻转方向 | 0.67（把两段都为负读成方向不一致） | 1.0 / 0.67 / 1.0 |

0.67 那次诊断正确（「方向与假设相反」），但判断翻转后仍过不了门槛，没有给修改。

## 真实研报试跑：中银证券《风格制胜 3：风格因子体系的构建及应用》（2025-06-06）

20 页 PDF，从东方财富研报中心下载，只在本地使用（`data/reports/`，不纳入版本管理）。

- **研报内容**：讲市值、估值、盈利、动量四类风格因子，以及它们与宏观、情绪、资金的关系。个股层面能用日线检验的只有「A 股 1 个月反转效应显著」（p1、p8、p9、p13）。
- **流程表现**：抽出 30~40 条观点。估值、盈利需要财务数据，风格轮动属于择时，都进了「不可检验」并写明缺什么数据。
- **结论**：20 日反转每次都被提出来，研究层判定它和因子库里的 20 日动量相关 -1.00，本身也不显著，淘汰；各种「红利资产反转」「新兴成长反转」变体按值去重跳过。研报没有产出新因子，登记表仍为空，这个结论本身是对的：风格研报讲的就是已知风格因子。
- **试跑中暴露并已修复的问题**：
  - 一次性生成假设时模型自己编了 6 个研报没提的因子 → 加忠实性核查。
  - 文本相似就跳过已淘汰方向，误跳了不同的因子 → 改成按因子身份跳过。
  - 输出过长被截断导致 JSON 解析失败 → 精简重试。
  - 一次性生成假设，每次挑的都不同 → 改成逐个观点判断。
  - 「A 股反转效应」被当成择时 → 提示里区分截面判断和择时。
  - 同一因子被写多遍 → 按值去重。
  - `groupby().rolling()` 索引多一层，模型修 3 次都没修好 → 报错附上因子代码行号，数据约定里写明这个坑。
- **解析改进后的回归**：核查节点变严后，一度把「20 日反转」剔掉了，理由是「研报没定义反转」，因为文字写的定义在别的段落。现在会给假设和核查节点附上全文最相关的 2 段原文，反转重新被提出来，结论与之前一致；token 因此升到约 7.5 万。
- **仍然存在**：假设生成和忠实性核查都有波动。同一份研报跑多次，「拥挤度增强动量」时有时无；有一次「小市值 = log(收盘价)」没被核查拦住，最后由研究层淘汰。一次运行约 110~140 次调用、4.5~5.5 万 token，耗时约 3 分钟。

## 真实研报试跑：国金证券《Alpha 掘金系列之四：基于逐笔成交数据的遗憾规避因子》（2023-03-01）

22 页 PDF，同样只在本地使用。这是一篇个股量价金工研报，但因子全部用逐笔成交数据定义（高于收盘价买入的成交量占比、小单、尾盘 30 分钟等）。

- **第一次跑**：4 个因子被判为「可检验」，写代码一层用日线自己编了近似去算，4 个都被淘汰。这个结论说的是近似，不是研报的因子，不能说明研报的因子无效。原来的检查只看字段名存不存在、变量是不是研报说过的，不看数据粒度够不够。
- **修复**：
  - 生成假设和忠实性核查都要求判断「能否只用日频字段精确算出研报定义的变量」。需要日内、逐笔或订单数据的判为不可检验并写明原因，不拿近似去测。
  - 另外，两段 RankIC 都为负但都不显著时，规则不再允许翻转方向，因为翻过来也是噪声。
- **修复后**：0 个假设，约 40 条观点全部进入「不可检验」，原因是需要逐笔、订单、日内分钟或持仓数据，或者是策略绩效描述。这是正确的结论：这篇研报的因子用日线无法验证。
- **公式和表格解析（第二轮）**：
  - **原问题**：公式是 PDF 数学斜体，分数被拆成多行；表格被压成一行，抽出了约 40 条观点，其中很多是绩效数字；部分原因写成「因子定义未给出」，实际是定义没读出来。
  - **解析结果**：20 张表格结构化（标题、列名、数据行），14 个公式带名字收拢（如尾盘因子的 `t∈[14:30,14:57)`），19 张图去掉。
  - **流程变化**：观点降到 12~17 条，原因能落到具体数据（「LCP 需要主动卖出成交价」），调用从约 140 次降到约 50 次、token 从 4.4 万降到 2~2.6 万。
- **程序兜底**：加了公式和原文上下文后，模型有一次又让 2 个日线近似的因子通过了「能否精确计算」的判断（把逐笔层面的 LCVOL 写成「下跌日成交量占比」）。现在由程序按数据粒度关键词兜底：逐笔、分钟、尾盘、委托、大小单、按买卖方向区分的成交，以及公式里的单笔成交下标，命中就判为不可检验。

## 已知局限

- 样本只有 2023 年以来约 3.8 年，训练段 2023-01~2025-04、验证段 2025-04~2026-01、测试段 2026-01~2026-10。训练段显著的低波动、上影线、低换手效应在验证段全部失效，单一验证段对风格切换很敏感。
- 没有 ST 的 5% 涨跌幅；行业在 2021 年申万新版发布前用最早一条记录回填；海通证券、中国重工（合并退市）没有行业和部分行情。
- 沙箱是子进程 + 资源上限，没有文件系统 / 网络隔离，生产环境应再套容器。
- 模拟账户是自己写的撮合，没有接券商或第三方模拟盘接口；不撮合集合竞价、不考虑成交量容量，滑点是固定比例。登记因子目前等权合成。
- 研报模块的输出有模型波动（见上面两节），重要研报建议多跑几次取并集。
- PPT 解析只在 Keynote 导出的文件上实测过（加上人工插入的表格）；PowerPoint / WPS 做的真实券商路演 PPT 还没试过，SmartArt、文本框里的图片文字都不解析。
- PDF 解析针对「图表 N：标题 … 来源：」这种文字排版的券商研报；扫描件、图片里的表格、公式的上下标结构（只保留了拆行后的顺序）都没有处理。
- 东方财富上公开的个股量价金工研报很少；日频可验证的研报需要自己提供 PDF。
- `--llm openai` 路径尚未实测。
- Qlib 接了数据、表达式计算、回测复核和模型训练（Alpha158 + 梯度提升树，用 Python 搭的 workflow，没有用 `qrun` 的 YAML 配置）；评估（中性化 RankIC、Newey-West、分段检验）仍是自己的实现。这台机器上 LightGBM 因缺 libomp 加载不了，实测用的是 sklearn 的同类模型。模型只在一个不到一年的测试段上验证过，没做滚动训练。自带 `--csv` 行情时不启用 Qlib。
- 续跑时，中断那一刻正在执行的节点会重做一次（比如正在等模型返回的「生成代码」）；节点里写经验库的操作因此可能重复一次，经验库会把近似重复的记录合并。
