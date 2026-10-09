# Static Forward V4 多机 inference 记录

## 2026-10-09 补充：滚动窗口与流式 pack

结论：**连续发射之后，足够并发的逐 token 通信与流式 pack 已非常接近；
不能再把上一轮串行 pack 的额外耗时直接算到流水实现上。**
32 owner 下，逐 token 的滚动窗口比相同实现的单段窗口快约3.5%–4.5%；
流式 pack 相比逐 token，pull 的8K慢1.5%、20K基本持平，push 快0.9%–1.8%。
这些都是独立通信实验，不是 PF FWD 加速；此次仍未修改生产算子。

### 主结果：32 owner，单位 ms

下表均为 **16 rank 的逐 sample 最大 total，再对两轮40次测量求平均**。
total 包含 pack、ready 和实际通信完成，不是仅计 enqueue，也不扣掉流式 pack。
两机各8 GPU、SM1620MHz、H4096/BF16、每 rank 在途请求上限16MiB，
沿用上轮相同的20组随机 token 数和随机行排列。

| 名义 tokens | 方式 | Q1 分批等完 | Q1 四段滚动 | Q16 整批 pack 后通信 | Q16 流式 pack + 四段滚动 |
|---|---|---:|---:|---:|---:|
| 8K | pull | 1.572 | 1.499 | 1.695 | 1.522 |
| 8K | push | 1.600 | 1.539 | 1.703 | 1.525 |
| 20K | pull | 3.700 | 3.544 | 4.022 | 3.545 |
| 20K | push | 3.796 | 3.636 | 4.040 | 3.572 |

与同轮整批 pack 的 total 相比，流式 pack 改善约10%–12%。
Q1 滚动窗口的有效单向吞吐约45.0–47.1GB/s/rank，已经接近大包传输。
20K pull 的0.014%差别不视为确定收益；8K push 的不足1%也不作强结论。

### 具体改动与同步

保留 `WEFTMOE_GIN_BENCH_MODE=burst` 复现旧实验；`pipeline` 增加以下对照。
所有候选仍是同一个 cooperative kernel 内的独立实验，没有 CPU 逐段调度。

| mode | 发射与完成确认 | pack 工作分配 |
|---|---|---|
| burst | 原版：发完一批，flush/wait，Put 再等本批入站计数 | 全128 CTA 先完成整批 pack |
| window，D=1/2/4 | 同一容量拆成 D 段，等最老一段完成后立即补发；其余段仍在队列内 | 与 burst 相同，整批 pack 串行 |
| staged，D=4 | 与 window 相同 | 每个 owner 对应一个独立 producer CTA，仍先 pack 完再通信 |
| stream，D=4 | 与 window 相同，增加分段 source-ready | 与 staged 完全相同的 producer 布局，与网络 owner 并行 |

- `C × B = 2048 tokens`，C/B 分别为4/512、16/128、32/64。
  每段 B/D tokens，每个 owner 最多保留 D 个完成句柄。
  Q1 每个请求8KiB；Q16 每个请求128KiB，尾包按实际行数。
  改 D **不改变每个 owner 分到的 token 集合**，只拆开其原来的 B 行批次。
- Get 用 `flushAsync + wait` 获取一个子批的完成/本地可见性。
  填满 D 段后，只等最老句柄，再立刻填入新段，不等整个窗口清空。
  NCCL 2.31.2 当前公开 Get 接口没有 local-counter 参数；本实验没有虚构一个
  Get completion counter，也没有依赖私有的非阻塞 request poll。
- Put 每个实际请求绑定 weak ready increment；滚动窗口用本地 flush 完成回收
  发射额度，最后等待全部入站请求的累计计数。这里的 RX 是完整数组，只有最后的
  总计数才证明全部 payload 到齐；**不能用部分累计计数推断某个 RX slot 已到齐**。
  `window D1` 单独控制了 Put 从“逐批等入站”变为“末尾等入站”的差异。
- stream 的 producer 按静态顺序 gather 一段，所有写线程完成 system release，
  CTA leader 再用本地 release store 发布 `pack_ready[owner]` 前缀。
  通信 owner acquire 后，Push 直接发 Put；Pull 则先把已生产前缀通过 GIN signal
  发给 peer，再等待 peer 的相应前缀，才发 Get。先发布后等待，避免双向 pull 互等。
  一个前缀只有一个顺序 producer，没有动态任务领用或跨 producer done_count。
- ready 发布和数据请求共用该 owner 的独占 GIN context；producer CTA 不调用 GIN，
  不存在两个线程并发操作同一独占 context。前缀和信号跨 sample 单调推进，
  每个 case 初始化一次；source/X、packed TX、RX 在 sample 内不覆盖。
- staged 与 stream 都只用 C 个 pack CTA。保留 staged 很重要：不能把 pack 从
  128 CTA 改成32 CTA 后的性能变化，全部归因于是否 overlap。

这是真正的 **pack/network 阶段并行 + 子批连续补发**，但仍不是生产版 ring：
X、packed TX、RX 均按本次最大实际 token 数预分配完整 plane；16MiB 是请求窗口，
不是总 buffer 大小。没有 RX slot wrap、消费 credit、GEMM、机内 fanout 或 reduce。
每轮开始的双向 rendezvous 和计时外全量校验，保护跨 sample 的 buffer 复用。
stream 的 `network_us` 包含并行 pack 的等待，不能叫“剔除 pack 的纯网络时间”。

### 分步收益与排除的混杂因素

1. **只改窗口深度**：C32 的 `window D1 → D4`，pull 的8K/20K为
   1.567→1.499 / 3.713→3.544ms；push 为1.594→1.539 / 3.779→3.636ms。
   D2 与 D4 的 total 差别均小于0.1%，本组不需要继续加深窗口。
   C4 的 Q1 仍受小请求发起并发限制，单纯增加流水深度没有明显收益。
2. **同一 pack 布局，串行改并行**：C32 的 staged→stream，8K pull/push 为
   2.070→1.522 / 2.073→1.525ms；20K为4.838→3.545 / 4.875→3.572ms，
   total 降低约26%–27%。owner 0 的首个网络请求发出后，producer 0 仍继续 pack
   约0.56–0.57ms / 1.33–1.36ms，证明 pack 并非提前做完后移出计时。
   这是该 producer 的时间戳诊断，不是全体 producer 的完成时间。
3. **pack 遍历小优化**：探索版把扁平 vector 下标改成 row/vector 两层循环，
   将每次 uint4 拷贝的运行时64位行号除法移出内层循环。在同一细条带布局的
   C32 相邻消融中，串行 producer pack 从约0.672/1.587ms降到0.553/1.294ms，
   改善约18%–19%；stream total 仅从1.526/3.549ms到1.517/3.543ms（pull），
   大部分拷贝成本本来就在重叠区间。这组是相邻构建证据，不分摊成正式 PF 收益。
4. **固定 owner 分工再复测**：初版按更细子批重新分条带，同时改变了尾批负载均衡。
   最终改为拆分原 owner 的批次、保持其 token 集合不变；上面所有主表均来自
   这轮重新测试。探索版单列保存，不把其全部差值当作流水收益。

### 少量 owner 的边界

下面三列都是 total，单位 ms；`window` 均为 D4。Q16 window 仍使用128 CTA
先 pack，Q16 stream 则只有 C 个 producer CTA。

| owners | 规模 / 方向 | Q1 window | Q16 window | Q16 stream |
|---:|---|---:|---:|---:|
| 4 | 8K pull | 4.438 | 1.707 | 4.348 |
| 4 | 8K push | 4.523 | 1.741 | 4.349 |
| 4 | 20K pull | 10.190 | 4.007 | 9.865 |
| 4 | 20K push | 10.376 | 4.095 | 9.866 |
| 16 | 8K pull | 1.502 | 1.692 | 1.541 |
| 16 | 8K push | 1.651 | 1.727 | 1.584 |
| 16 | 20K pull | 3.541 | 3.993 | 3.555 |
| 16 | 20K push | 3.787 | 4.073 | 3.639 |

C4 时 pack 仍很有价值：即使先 pack 再传，total 也比 Q1 快约2.5倍。
但只有4个 producer CTA 的流式版本，其串行 pack 本身就需4.24/9.74ms，
生产速度成了瓶颈，不能靠 overlap 消掉。这说明 **pack 并行度不能机械地等于
通信 owner 数**，不是“异步 pack 天生更慢”；尚未扩展多个 producer 服务一个 owner。
C16 pull 不 pack 已接近最优，而 push 的流式 pack 约快4%。

对 PF 的当前建议：dispatch 可以先保留“不额外 pack 的逐 token pull + 滚动窗口”
作为简单候选；如果可用通信 owner 很少，必须同时考虑聚合包。
return 若 FC2 能直接产出连续 TX，可用 Q16 network 阶段估计不付 gather 的通信潜力。
最终还要在真实 kernel 中比较通信/pack 占用的 SM 与 GEMM 竞争，本实验不能决定
最优 CTA 配额，也不能代替按节点去重、非均衡 fanout 和有消费 credit 的 ring 验证。

### 验证、冻结数据与复现

- 正式 C4/C16/C32 共 **96,000 次 rank/sample 全量 payload 校验通过**，含随机
  尾包；另有两 GPU、T≈1/65/1024 的720次校验，覆盖空 owner 和不足一窗。
- 正式三轮共672次 GPU 时钟观测全部 SM1620MHz，活跃显存时钟全部3201MHz；
  所有 rank 为 NCCL 2.31.2 GDAKI。source、RX 初始化、分配/注册和校验均在计时外。
- [流水消融冻结数据](../benchmarks/gin_pack_pipeline_20261009.json) 保存正式
  逐样本 rank-max、探索版汇总、时钟/校验统计和源代码/binary/log SHA256。
  完整逐 rank 日志为 `build/gin_pack_pipeline_final_c{4,16,32}.log`。
- 两节点相同 binary SHA256：
  `60919d30de1c6b8fa1bb495fa51be87b589a94678cf61e64b2e596c2ac310669`。
- 使用下节同一编译/启动命令，增加 `WEFTMOE_GIN_BENCH_MODE=pipeline`；
  C/B 仍分别设4/512、16/128、32/64。默认 burst 模式保留旧 Q1/4/16/64 矩阵。

## 2026-10-09：跨机逐 token 与 pack 的最小消融

结论：**pack 明显降低低并发下的网络请求开销，但本例并非必须先 pack。**
32 个通信 owner 时，逐 token 的纯通信已接近打包版；本实验把整批 pack
串行计入后，不 pack 反而更快。不能把这个结果外推成“流式 pack 与通信折叠也无收益”，
也不能把纯通信收益当作 PF FWD 收益。此次不修改 PF/DT/BWD 生产实现。

### 测量合同

- 两机各 8 张 L20X，8 对同 local rank GPU 同时双向通信，不是 2 机 EP8。
  NCCL 2.31.2，所有 rank 实报 GDAKI（railed_gin_type=3）。
- H=4096、BF16 等宽原始数据，一 token 8192B。8K/20K 为名义每 rank
  传输 token 数，逐 sample 采样正态分布（标准差 10%）；正式样本均值分别
  8453 / 20400.15，范围 6706–10158 / 16782–26038。
- 所有变体重放相同 token 数、相同每 rank 随机行排列；每个 sample 重新打乱行。
  这是隔离传输粒度的一对一随机行 gather，不含 MoE top-k 展开或节点去重。
  各 rank 的 token 数相同，不将其描述为完整不均衡 MoE workload。
- Q=1：每次 Get/Put 从原始 X 的随机 token 行搬 8KiB，直接落到连续 RX 行。
  Q=4/16/64：先用 128-bit vector gather 到连续 TX，再按 32/128/512KiB 发请求。
  接收布局、有效字节数完全相同，两者都不需要 unpack。
- 固定总在途上限 2048 tokens/rank=16MiB；contexts × 每 context batch
  分别为 4×512、16×128、32×64。各 context 独占固定任务，无动态领用。
  Q 改变时不增加在途字节上限，也不多传 payload。GIN queue depth=1024。
- Get 每批用 flushAsync(peer)+wait 确认本地可见；Put 每个请求绑定 weak
  ready signal，每批 flush 确认本地读完，并等待自己的相应接收计数。
  因此 pure network 包含真正可消费的完成确认，不是仅计 enqueue。
- 原始 X、packed TX、RX 一次预分配/注册，计时不含分配、初始化、路由数组构造、
  NCCL bootstrap、校验及最终销毁。Q=1 也预留相同的三个 plane，排除容量差异。
- 单个 cooperative kernel 内分阶段记录 globaltimer。network 从 source-ready
  rendezvous 后开始，到全部本地 owner 完成；total 包含 pack、相同的 source-ready
  rendezvous 和 network。pack **没有**与本批通信重叠，也没有 GEMM。
  pack_us 含其局部 grid joins（Q=1 约 2.5us 的空阶段）。
- 每个 case 5 warmup + 20 measured，第二轮逆序 Q 和 Get/Put。下表为
  **每 sample 先取 16 rank 最大值，再对两轮共 40 samples 求平均**，单位 ms；
  不是早期固定负载的 p50。各 stage 的 rank-max 可能来自不同 rank，不能相加。
- 每轮先 poison RX，通信后检查全部 32-bit words，不只比较抽样或 hash。
  最终三种 contexts 合计 38,400 次 rank/sample 全量校验通过；每组 160、
  共 480 次 GPU 时钟观测均为 SM1620MHz，活跃显存时钟均3201MHz。

### 32 contexts：完整通信及额外 pack 代价

| 名义 tokens | 方式 | Q1 network | Q16 network | Q1 total | Q16 total | pack 后 total 变化 |
|---|---|---:|---:|---:|---:|---:|
| 8K | Get / pull | 1.554 | 1.483 | 1.566 | 1.705 | +8.9% |
| 8K | Put / push | 1.584 | 1.487 | 1.597 | 1.711 | +7.2% |
| 20K | Get / pull | 3.700 | 3.505 | 3.712 | 4.014 | +8.1% |
| 20K | Put / push | 3.756 | 3.543 | 3.769 | 4.052 | +7.5% |

打包后纯通信约快 4.6%–6.1%，但额外 pack 约 0.214ms / 0.506ms，
使 total 慢 7.2%–8.9%。Q1 的有效单向 payload 吞吐约 43.7–45.2GB/s/rank；
Q16 约46.6–47.7GB/s/rank。这里不把双向流量乘二冒充单向带宽。
Q16 增到 Q64 的纯通信变化在约1%以内，没有继续增加包长的明显依据；
Q4 结果及全部逐样本数据保存在 JSON。

### 并发度消融：不能只测少量 issuer 就判定必须 pack

下表每格为 **Q1 total → Q16 total**，单位 ms；在途容量均为16MiB/rank。

| contexts | 每 context batch tokens | 8K pull | 8K push | 20K pull | 20K push |
|---:|---:|---:|---:|---:|---:|
| 4 | 512 | 4.352 → 1.776 | 4.536 → 1.811 | 9.990 → 4.203 | 10.415 → 4.283 |
| 16 | 128 | 1.609 → 1.702 | 1.706 → 1.736 | 3.745 → 4.012 | 3.908 → 4.119 |
| 32 | 64 | 1.566 → 1.705 | 1.597 → 1.711 | 3.712 → 4.014 | 3.769 → 4.052 |

4 contexts 时，pack 的 total 加速约2.4–2.5倍；Q1 的纯通信只有约15–17GB/s。
32 contexts 时，Q1 通过更多独立 issuer/QP 达到约44–45GB/s，省掉额外 pack 后
反而占优。这支持“小请求的发起/进度并发是重要因素”，但不能仅凭本实验进一步
区分 WQE 构造、doorbell、QP 执行和完成等待各自的成本。

后续多机实现的决策边界：

1. Dispatch 可优先试按节点去重的逐 token pull，不必预设整批 pack 是必要步骤。
   同时必须给足通信并发；不能直接把本实验的32个owner照搬成PF的最优资源配额。
2. 如果 FC2 本来就能直接产出连续 return TX，便没有这里约0.2/0.5ms的额外 gather，
   聚合 Put 仍值得保留；本实验没有否定这种“生产时即排好布局”的方案。
3. 尚未测试流式 packet pack 与网络/GEMM 的 overlap、节点内 fanout、ring 消费
   credit、跨更多节点和真实 FWD 资源竞争。不得据此承诺 PF 的端到端收益。

### 代码、数据与复现

- 独立最小实现：[gin_pack.cu](../benchmarks/gin_pack.cu)。
- 启动及聚合：[run_gin_pack.py](../benchmarks/run_gin_pack.py)，复用既有双机
  bootstrap、SSH 和进程清理，不链接 WeftMoE 主算子。
- [冻结数据](../benchmarks/gin_pack_20261009.json)：含每项 rank-max
  逐样本数据、随机 token 数、所有 rank 的 backend 证据、校验/时钟统计及文件哈希。
  完整逐 rank 日志为 build/gin_pack_final_c{4,16,32}.log。
- 两节点实测 binary SHA256：
  `0c924eb5ffa1aded4a72194993a5a816a56735bee631d67bfe9ac3ea079e767a`。
  另有单对 GPU、T≈1024、C4 的快速全量校验，不用于正式性能结论。

在 repo 根目录编译，并将同一 binary 放到第二节点的同一路径；两端先固定时钟。
以下以 C32/B64 为例，改为 C16/B128 或 C4/B512 即可保持相同的总在途上限：

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 --expt-relaxed-constexpr \
  -arch=sm_90a -Xcompiler=-Wall,-Wextra benchmarks/gin_pack.cu \
  -lnccl -o build/gin_pack
scp -P 2222 build/gin_pack \
  root@10.10.19.233:/storage/openpsi/codes/wanghaitao.wht/project/ops/WeftMoE/build/gin_pack

env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64 \
  NCCL_IB_GID_INDEX=3 WEFTMOE_GIN_BENCH_CONTEXTS=32 WEFTMOE_GIN_BENCH_BATCH=64 \
  /opt/.venv/bin/python benchmarks/run_gin_pack.py \
  --binary "$PWD/build/gin_pack" --remote-binary "$PWD/build/gin_pack" \
  --remote root@10.10.19.233 --port 2222 --gpus-per-node 8 \
  --socket-ifname eth0 --enable-ib \
  --nccl-lib-dir /usr/local/cuda/compat:/usr/local/cuda/lib64 --timeout 240
```

## 以下为 2026-09-22 起的历史实现与测量

日期：2026-09-22；2026-09-23 补充优化分步记录与 V3 对照。协议与首版边界见
[`static_forward_v4_design.md`](static_forward_v4_design.md#60-复审进度容量与首版实现边界)。

后续的 prefill 双入口、epilogue 消融及跨机 Return/GEMM 重叠分析，见
[`static_forward_v4_prefill_record.md`](static_forward_v4_prefill_record.md)。

拓扑更正：用户要求的主矩阵是 **单机 EP8（1×8）和双机 EP16（2×8）**。
此前误将 EP8 测成两机 2×4；本篇保留该组数据作为额外实验，不代表单机 EP8。
单机 EP8 的 1620 MHz 锁频补测见
[`static_forward_v4_optimize_record.md`](static_forward_v4_optimize_record.md)。

本篇冻结版本结论（两节点全部 GPU 核心锁频 1620 MHz）：对照 V3 inference fallback
及已有多机调优参数，V4 在 1/64 tokens 加速 **3.86–4.35×**，但 8K/20K 延迟
增加 **15.1–34.2%**。功能与数值门禁已通过，**多机 prefill 性能目标尚未达成**，
不能称为全面优于 V3。完整锁频矩阵和比较边界见下文。

## 优化路径与逐步收益

沿用单机记录的证据等级：A = 同条件反转顺序的交错 A/B，B = 相邻构建消融，
C = 多项同时变化的里程碑 bundle。所有 token 数均为每 rank；延迟单位 us。
未独立测量的改动不分摊收益，也不把单机收益搬到多机。

| 步骤 / 提交 | 改动及解决的问题 | 多机收益与证据边界 |
|---|---|---|
| 0 / `b4538e0` | 复用单机的 FC2 连续 128-byte 写、32-Mblock 静态 wave、M128 readiness 直接索引 | 已包含在第一版多机中。单机 8K/20K 相对复审基线改善 24.9%/36.0%（B）；没有多机独立消融 |
| 1 / `d69f6b6` | 放弃 bounded ring 协议，改用完整 invocation arena；区分 NIC 本地读完、远端可见、consumer 用完 | 设计修正，无独立性能数值。避免 TX 等 credit 时阻塞同 owner 的 RX 进度，以及错误复用 TX buffer |
| 2 / `1154748` | 每个 rail 固定 CTA owner；原始 X 每远端节点一份，expert-sorted route/prefix 驱动节点内 pull；移除 V3 动态 claim/packet/fanout/credit 控制路径 | EP16/64 首个正式 graph 基线 **191.47**（C）。不是相对 V3 的加速比 |
| 3 / `1154748` | FC2 直接写源 rail 的独立 Return TX；按目的节点 expert 区间紧凑回传，combine 直接读 RX；共享 route geometry、pull、stable combine order | 减少额外 scatter 与控制元数据，已包含在 191.47 中；没有单项计时 |
| 4 / `1154748` | serialized graph 去掉 GIN reuse event，保留 eager 跨 stream 保护 | 原型 1 kernel + 1 event-wait 变为 **1 kernel、0 event/copy**。原型 EP16/1=168.90 仅 2 samples、batch=1，不与正式 batch=20 值计算收益 |
| 5 / `c3610bb` | Combine 后跨机 barrier 改为本 GPU grid join；GIN contexts 由统一 capacity 确定，并缩小实际分配 | EP16/64 **191.47 → 173.31**，−18.16 / **−9.5%**（C，相邻构建）。两项同批变化，不分别归因 |

步骤 2–4 属于一次多机大重构，没有保存每步独立可比较的 timing。
容量检查改为真实用户 buffer 大小、允许空 rank 的空 gate，以及 collective 拒绝后
恢复复用属于正确性修复，不计作性能优化。各规模最终数据见后文完整矩阵。

与旧路径的关键区别不是简单减少几个 CTA：静态 residue 分配替代动态 GEMM 任务领取，
full-invocation staging 替代依赖 credit 的逐包进度，返回数据直接落到 combine 可消费
的位置。V3 SPSC 已经有固定 TX/RX channel owner，不能把“去掉 GIN context 锁”重复
记成 V4 的新贡献；V4 进一步把通信收敛为 1–8 个 rail owner 的连续 Put 和阶段 fence。
代价是更大容量预留，以及跨机 Dispatch/Return 仍为显式阶段，未与 GEMM 流水重叠。
不能据此推断所有模型/路由都更快。

## 实现门禁

- 本篇的两机实验为额外 EP8（2×4 GPU）和主矩阵 EP16（2×8 GPU），不是单机 EP8。
- H=2048，I=512，64 experts，top-k=4，BF16。
- 1、64、8K、20K 均为每 rank 的 token 数；8K=8192，20K=20480。
- 两种 EP 的四个规模均通过 PyTorch 独立参考的 eager 和 graph 校验。
  参考保留 FC1 BF16、SwiGLU/gate FP32 后 BF16、FC2 BF16、按 `(expert,slot)`
  FP32 combine 的数值合同。最大绝对误差 7.63e-6，阈值 rtol=0.02/atol=2e-5。
- 两种 EP 均检查不等长/空输入 rank、所有 route 重复指向最后一个 expert、四轮
  X/routes/gates 变化、eager 与 graph 交错、双层 graph 共用通信 workspace，
  以及容量不足时集体拒绝、随后复用同一 workspace 恢复计算。
- EP8 20K 的所有输出哈希与单机完全相同；EP16 1/64 的前 8 个 rank 也与
  对应单机输出完全相同。fixture 的输入和 expert 权重不依赖 EP 划分。
- 原型图中的 GIN reuse event 已移出 serialized graph hot path；每 rank
  一个 kernel node，零 memcpy/event node。eager 保留跨 stream 的 event 保护。

测试入口为 `tests/torchrun_static_inference.py --tokens 1 64 8192 20480
--edge-cases`，由两节点 torchrun 启动，各节点 `--nproc-per-node=4/8`。
环境需设置 `PYTHONPATH` 为 repo 根目录，`WEFTMOE_TENSOR_EXTENSION` 指向同一
native library 构建的 `_runtime*.so`，本机 RDMA 配置使用
`NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0`。

初版验证日志：`/tmp/weft-v4-parity-ep16-v3-{local,remote}.log`（四个规模通过，
随后空 gate 的旧接口检查报错）及 `/tmp/weft-v4-parity-ep8-v4-{local,remote}.log`
（零长度 gate 修正后全部通过）。零长度 gate 现在与零长度 route 一样是合法输入。

最终日志为 `/tmp/weft-v4-parity-ep{8,16}-final-{local,remote}.log`，两种 EP
分别完成 176/352 次 rank/case 校验，所有进程正常退出；容量拒绝后的恢复也已通过。
额外运行原有 `tests/torchrun_tensor_runtime.py`（单机 EP8、8 tokens、64 experts），
8 rank 的 Forward/Backward parity、双层 graph、销毁/offload 均通过；训练仍走 V3。
日志为 `/tmp/weft-v4-training-regression.log`。

## 2026-09-22 历史测量合同与矩阵

本节保留当日原始记录，不属于后文的 1620 MHz 锁频对照；不要跨表计算加速比。

冻结的构建参数位于 `configs/static_inference_20260922/`，不依赖临时 JIT cache。
在 repo 根目录构建（使用 CUDA 12.9、CUTLASS、NCCL 2.31.2）：

```bash
export NVCCFLAGS="-I$PWD/configs/static_inference_20260922 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1"
python3 build_native.py cuda-test --jobs 8 --output build/static_v4_network/test_cuda_notify
python3 build_native.py lib --jobs 8 --output build/static_v4_network/libweftmoe.so
python3 tools/build_tensor_runtime.py --library build/static_v4_network/libweftmoe.so
```

默认产品构建/其他 H/I 应使用自己的 tuning，不把这个测量点冒充所有模型的最优配置。

裸 CUDA Graph 使用 CUDA event，10 个 warmup sample + 20 个 measured sample，
每 sample 连续 replay 20 次；每轮先取所有 rank 的最大耗时，再取 p50。
包括 route exchange、输入 staging、GIN Dispatch、节点内 pull、计算、Return、
combine 与清理。不包括进程/NCCL 初始化和 graph capture。

`tests/run_multinode_nccl.py --scenario gin_forward_bench --gpus-per-node=4/8`
使用以下环境：

```text
WEFTMOE_BENCH_EXPERTS=64
WEFTMOE_P2P_TOP_K=4
WEFTMOE_P2P_BENCH_TOKENS=<1|64|8192|20480>
WEFTMOE_FORWARD_BENCH_ROUNDS=30
WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY=1
WEFTMOE_FORWARD_GRAPH_BATCH=20
```

初版 EP16/64 的 p50=191.47 us；初测 EP8/20K=9206.34 us 只用了两个样本，
不能作为正式矩阵值。

当日最终构建的历史完整矩阵如下，单位 us，均为上述 graph rank-max p50：

| 拓扑 | 1 token | 64 tokens | 8192 tokens | 20480 tokens |
|---|---:|---:|---:|---:|
| 单机 EP8（最终回归） | 58.12 | 68.16 | 1480.64 | 4583.34 |
| 两机 EP8（2×4） | 145.61 | 186.95 | 3752.12 | 9201.61 |
| 两机 EP16（2×8） | 154.48 | 173.31 | 3703.26 | 9191.01 |

8 个 case 的前 8 rank 哈希全部与对应单机基准完全一致，所有 rank 的图都只有
1 kernel、0 memcpy、0 event。原始日志为
`/tmp/weft-v4-network-final-gpus{4,8}-t{1,64,8192,20480}.log`。
这些 2026-09-22 数值不是与旧 V3 多机路径的交错 A/B。
12 个 case 的完整 rank/sample、图节点数、输出 hash 及 native binary SHA256
已保存到 [`../benchmarks/static_inference_20260922.json`](../benchmarks/static_inference_20260922.json)。

## 2026-09-23：V3 / V4 同条件对照（核心锁频 1620 MHz）

对照的是当前提交 `c3610bb` 的 V3 inference fallback，不是训练模式，也不是用
此前其他 E/top-k 的历史时间作分母。V3 设置 `WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`，
使用仓库已有的 `communication_spsc_20260916/qwen.hpp` 多机参数；构建入口冻结在
`configs/static_inference_v3_20260923/`。V4 沿用上述最终构建和参数。
V3 参数曾在 H2048/I512 的多机路径验证，但这次没有针对 E64/K4 和四个长度重新
autotune；因此结论适用于这组公开配置，不宣称战胜所有可能的 V3 调参结果。

V3 的 FC1/FC2 N tile 为 128/256、K64、stages 4/2、FC2 claim=2，通信域为
16 sender + 116 receiver/local；V4 为 N64/N128、K128、stages 3/3，静态 wave=32。
二者均为 132 CTA、384 threads、cluster=2。比较保留各自参数，是实现与配置的
整体对照，不是仅替换 scheduler 的单变量实验。

每个 case 按 **V3 → V4 → V4 → V3** 启动四个独立 session，各自 10 个 warmup
和 20 个 measured sample，每 sample 20 次 graph replay。打开
`WEFTMOE_BENCH_SYNC_RANKS=1`，在每个 sample 的计时区间外同步所有 rank；
先逐 sample 取 rank-max，再合并同一版本两个 session 的 40 个 sample 取 p50
（排序后第 20 个）。这是独立进程的反序复测，不冒充同一进程逐 iteration 切换。
正式计时不并发编译或其他 GPU 测试。按要求，两节点全部 16 GPU 使用
`nvidia-smi -lgc 1620,1620` 锁定核心时钟，并设置 `-ac 3201,1620`；显存 application
clock 保持 3201 MHz，未另行锁显存频率，driver=570.148.08。以 200 ms 间隔记录
两节点各 GPU 的实际 SM/graphics 时钟及利用率，检查负载期间是否维持 1620 MHz，
而非只检查 application-clock 配置。
此前 application-clock=1980 MHz、未锁频的预跑不纳入本节结果，也不与本轮样本合并。
前文 2026-09-22 的里程碑数字保留为历史记录，不拿它们与本轮锁频结果计算优化收益。

V3 graph 包含 Notify + Forward，共 12 nodes：2 kernel、3 memcpy、6 event-record、
1 event-wait。V4 将 route/notify 融入 Forward，为 1 kernel、0 memcpy/event。
双方都测完整 graph，不人为从 V3 耗时中减去辅助节点；也不把 graph p50 与
最后一次 replay 的 device-kernel span 混用。固定输入/路由/权重在所有 session
中复用，输出以 2026-09-22 已通过 PyTorch 参考的 V4 全 rank 哈希为门禁。
每个 case 独立建 pool，token capacity 按该 case 配置；不是让同一个 20K capacity
pool 跑所有 decode 长度的测试。数值边界用例已验证，但本表不代表热点路由的性能。

### 锁频结果

单位 us；加速比 = V3/V4，延迟变化 = `(V4/V3 - 1) × 100%`，正数表示回退。

| 拓扑 | tokens/rank | V3 p50 | V4 p50 | 加速比 | 延迟变化 |
|---|---:|---:|---:|---:|---:|
| EP8（2×4） | 1 | 621.90 | 160.96 | 3.86× | -74.1% |
| EP8（2×4） | 64 | 793.43 | 199.48 | 3.98× | -74.9% |
| EP8（2×4） | 8192 | 3359.00 | 3939.05 | 0.85× | +17.3% |
| EP8（2×4） | 20480 | 7260.68 | 9614.13 | 0.76× | +32.4% |
| EP16（2×8） | 1 | 680.49 | 169.23 | 4.02× | -75.1% |
| EP16（2×8） | 64 | 824.51 | 189.52 | 4.35× | -77.0% |
| EP16（2×8） | 8192 | 3355.23 | 3860.82 | 0.87× | +15.1% |
| EP16（2×8） | 20480 | 7105.66 | 9537.37 | 0.75× | +34.2% |

32 个 session 全部完成，384 次 rank/output hash 检查全部与已验证参考一致。
03:56:26–04:08:06 UTC 的核心时钟采样共 55,928 条，全部为 SM=graphics=1620 MHz；
每张 GPU 均有负载样本，负载样本的显存频率均为 3201 MHz。测试后两节点仍保持
1620 MHz 锁频，没有恢复到 1980 MHz。

完整的各 session p50、rank/sample、图节点数、hash、构建 SHA256，以及逐 GPU
时钟核验汇总保存于
[`../benchmarks/static_inference_compare_1620_20260923.json`](../benchmarks/static_inference_compare_1620_20260923.json)。
原始日志为 `/tmp/weft-1620-abba-20260923-ep{8,16}-t{1,64,8192,20480}-{trial}-{v3,v4}.log`；
时钟日志为 `/tmp/weft-1620-clocks-{local,remote}-20260923.csv`。
该 JSON 另存了 2026-09-22 初版 EP16/64 的原始样本，明确放在不属于锁频对照的
historical 字段下，以保留 191.47 → 173.31 的里程碑证据。

### 结论与下一步

- decode 的静态任务、fused Notify、graph 节点精简确实带来整体收益，但这不是
  将整个 3.86–4.35× 分摊到单项优化的依据。
- prefill 回退在反序复测和 1620 MHz 锁频下仍存在。代码可确认 V4 的跨机 Dispatch、
  GEMM、Return 有阶段边界，而 V3 保留流式通信；同时两者 N tile、stages 和任务
  粒度不同。**没有单变量消融，不能将回退完全归因于同步或缺少通信重叠。**
- 后续应先比较 prefill 适用的 GEMM tile/wave 参数，再隔离测量跨机通信与 GEMM
  重叠的收益；保持静态任务与 full-arena 的进度保证，不重新引入已经否决的
  TX/RX 同 owner 等 credit 环。node-deduplicated X staging 仍是待实现项。
- 本次只补测和记录，生产调度未改，也未增加按长度自动切换 V3/V4 的策略。
  上述 prefill 回退仍未修复。

### 复现

V3 构建命令如下，V4 使用前文的构建命令：

```bash
NVCCFLAGS="-I$PWD/configs/static_inference_v3_20260923 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1" \
  python3 build_native.py cuda-test --jobs 8 \
  --output build/static_inference_v3_20260923/test_cuda_notify
```

运行时使用前文六项 benchmark 环境变量，额外设置 `WEFTMOE_BENCH_SYNC_RANKS=1`。
V3 设置 `WEFTMOE_DISABLE_STATIC_FORWARD_V4=1`；V4 **unset** 此变量。
launcher 示例为 V3 / EP16；测试 V4 时替换 binary，EP8 则设置 `--gpus-per-node 4`：

```bash
test_binary="$PWD/build/static_inference_v3_20260923/test_cuda_notify"
python3 tests/run_multinode_nccl.py \
  --binary "$test_binary" --remote-binary "$test_binary" \
  --remote root@10.10.19.233 --port 2222 --gpus-per-node 8 \
  --scenario gin_forward_bench --socket-ifname eth0 --enable-ib --timeout 240
```

本机另需 `LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64` 和
`NCCL_IB_GID_INDEX=3`。原 V4 冻结参数的 `WEFTMOE_FWD_RDMA_CTAS=0` 关闭了旧多机
入口，不能只在那个 binary 上设置 disable 开关就当作有效的 V3 多机对照。

## 首版后的同步精简（2026-09-22 历史证据）

初版每轮有三次跨机 barrier：Dispatch 发布、Return 发布、Combine 后复用。
最终版只保留前两次，末尾只做本 GPU grid join。其复用证明依赖不相交的 arena，
不依赖两个节点或 rank 的相对速度：

| arena | 上一轮最后的使用 | 下一轮首次覆盖前的边界 |
|---|---|---|
| source replicas | 节点内 pull 与 Return extent 读取 | 上一轮 Return barrier 已保证所有 peer 完成 Dispatch 消费并进入 Return |
| Return TX | 本节点 NIC 读取 | 下一轮 Dispatch barrier 保证所有接收 rank 已结束上一轮 Return/Combine |
| Return RX | 本 rank Combine | 下一轮 Dispatch barrier 必须等本 rank 结束 Combine 并进入下一轮，之后才可能有新 Return |

因此快 rank 可以先准备下一轮的 source replicas，但不能越过 Dispatch barrier
提前覆盖仍被消费的 Return TX/RX。每个 context 仍独立使用带 Put fence 的 rail
barrier，节点内 barrier 汇合所有 rail 的可见性；没有用 context 0 signal 代替
其他 context 的 fence。

GIN context 数按对称 capacity 推导：`min(8, ceil(max_route_return_bytes / 256KiB))`。
本矩阵的 1/64/8192/20480 分别为 1/4/8/8 个固定 CTA owner。不再为仅需这些
owner 的 inference pool 分配 V3 的 133 个 context。训练和显式 V3 fallback
仍沿用自己的 context 配置。这里使用 capacity 而非本 rank 的实际 token 数，
避免不等长输入让不同 rank 进入不同数量的 barrier。

EP16/64 从初版 191.47 us 降至 173.31 us，约改善 9.5%；这是末尾 barrier 与
context 精简的相邻构建 bundle，不将收益进一步拆成未经独立 A/B 的单项。

## 尚未覆盖与已知限制

- runtime 循环和 offset 支持任意节点数，没有两机特判；仅两机有实测证据。
- 跨机 Dispatch/Return 与 GEMM 之间有阶段边界；节点内 pull 与 GEMM 重叠。
- source X 当前向每个远端节点发送一次，即使该节点没有被路由到。后续可引入
  node-deduplicated compact staging；不能把该优化尚未实现写成已有收益。
- 完整 invocation arena 按公开容量预留；没有 ring credit、动态任务领取或丢 token。
- 扩展回归中 59 个 Python 测试和 86 个 subtest 通过，另有 8 项旧 autotune
  断言失败（候选数量、共享内存常数、宏枚举、incumbent 名称）。相关工具、配置和
  测试与基线 `8f3fa57` 没有差异；本轮不通过改写这些旧断言掩盖问题。
  去掉这两个已知过期的测试文件后，针对 unified resource、runtime settings、
  buffer validation、launcher security 的复测为 30 passed / 62 subtests passed。
