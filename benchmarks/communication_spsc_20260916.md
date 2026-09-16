# GIN 通信与同步改造记录（2026-09-16）

实现分支 `communication_spsc`。最终实现为 `360a6dc`，代码与 `1af6b79` 相同；
`235484c` 的 Return ready bitmap 实验因 EP8/8K 超时已完整撤销。
对照为本次改造前的 `0fd0d45`（在本分支通过 `284ac26` 保存），不是 main 或 Flex。

## 保留的实现

- 通道的 TX/RX 所有权静态划分，删除跨 CTA GIN context lease/锁。
  sender 的打包、提交状态放在 CTA shared memory；全局只有 sender 完成标志。
  FWD/BWD 复用同一个发送流程。
- 每个 `(context, arena, channel, physical slot)` 使用独立 local-completion counter。
  操作开始从 `readCounter()` 获取基准；每次 PUT 附带 `WeakCounterInc`。
  slot 复用必须同时满足自己的本地完成目标与远端 credit，不把任意完成计数视为前缀。
  比较按 56 位 rolling counter 处理；退出时也逐 slot 检查本地完成。
- PUT 的 ready signal 与其 payload 绑定。接收方 acquire ready 后搬运，完整等待 TMA 完成，
  再通过 GIN 归还该 slot 的 credit。发送端不逐包 flush；接收端退出时仅 flush 实际使用的 peer。
- NVLink 保留 push 模式：TMA 完整完成、发送线程同步、release 发布通知、接收 acquire、归还 credit。
  没有 pull 或空闲 GEMM worker 辅助路径。
- Return 行队列保留逐行 release 发布与独立的 32 行完成计数；没有等待其他 producer 的全局 confirmed prefix。
  本地搬运用非阻塞 CAS 领取工作，避免静态工作划分产生负载不均；这不是 GIN context 锁。

默认每个远端节点 64 个 Dispatch channels，每个远端 rank 2 条 Return streams，
每包 64 行、ring 深度 8。这些均为通用可调参数，没有 model/shape 分支。
EP16、132 CTA 配置下，16 个 CTA 的通信 warps 负责 TX，其余 116 个负责接收/NVLink；所有 CTA 保留 GEMM。
请求 contexts 数为 `max(legacy 所需数量, 1 + SM数)`（这里请求 133），映射使用 backend 实际返回数量。
context 0 留给控制操作；数量不足时循环共享，统一使用 GIN 的 GPU-scope 原子仲裁。
EP32 的 peer/channel 映射没有“只允许一个远端节点”的限制，但本次未做 EP32 GPU 验证。

复现配置：[Qwen](../configs/communication_spsc_20260916/qwen.hpp)、
[Flash](../configs/communication_spsc_20260916/flash.hpp)。Flash 保留已验证的 GEMM/staging 资源参数；
直接套用 Qwen 的 staging 数量超过 shared memory 上限。通信协议相同。

## 测量口径

两节点各 8 GPU，SM90a，CUDA 12.9、NCCL 2.31.2，锁定 1620 MHz。
记录中的 busy clock 采样为 1620 MHz。没有并行运行 GPU 测试，也没有在最终计时期间编译。
Qwen fixture 为 H=2048、I=512、E=256、K=8；长度是每 rank 的 tokens。
取各轮最慢 rank 时间的中位数。kernel 列不含 tensor 申请；wall 列包含固定 tensor 申请、复制、notify 和执行。
训练中复用的 pool、RDMA contexts、注册窗口等一次性建立时间不计入。
纯通信构建使用 `WEFTMOE_BENCH_TRANSPORT_ONLY=1`，仍有模拟计算阶段的调度开销，不能当作完整 GEMM 性能。

完整数据、构建 cache keys 和采样信息见 [JSON](communication_spsc_20260916.json)。
原始日志在 `results/communication_spsc_20260916/`。本次未重新运行 Flex，不据此声称相对 Flex 的收益。

## 纯通信结果

时间单位 ms，均为 **旧版 → 新版**。

| EP | 每 rank tokens | Dispatch | Return | 整个 kernel | 含 tensor 申请的 wall |
|---|---:|---:|---:|---:|---:|
| 8 | 8K | 0.871 → 0.897 | 0.817 → 0.820 | 1.790 → 1.821 | 2.679 → 2.685 |
| 8 | 64K | 6.671 → 6.909 | 5.768 → 5.771 | 13.231 → 13.329 | 14.442 → 14.590 |
| 16 | 8K | 1.022 → 1.141 | 3.171 → 3.165 | 4.390 → 4.430 | 5.794 → 5.820 |
| 16 | 64K | 6.676 → 7.688 | 23.241 → 23.503 | 29.980 → 30.984 | 31.854 → 32.836 |

以下为有效 payload 带宽，已排除 header/metadata；EP8 计 NVLink，EP16 仅计 RDMA payload。
它是整个相应阶段的平均值，不是链路瞬时物理带宽。

| EP | tokens | Dispatch GB/s，旧 → 新 | Return GB/s，旧 → 新 |
|---|---:|---:|---:|
| 8 | 8K | 269.8 → 261.9 | 287.6 → 286.3 |
| 8 | 64K | 281.7 → 272.0 | 325.8 → 325.6 |
| 16 | 8K | 32.8 → 29.4 | 42.3 → 42.4 |
| 16 | 64K | 40.2 → 34.9 | 46.2 → 45.7 |

## 完整 FWD 与正确性

Qwen EP16/8K retained-state 完整 FWD：kernel **5.035 → 5.372 ms（慢 6.7%）**，
wall **6.578 → 6.774 ms（慢 3.0%）**。因此本改造不能称为全面性能领先。
纯通信的 Return 接近旧版，剩余差距在 EP16 Dispatch 和与 GEMM 并发执行时仍明显。

- Qwen 与 Flash FWD：EP8/EP16 重分布及 129/258 token batch 前缀逐字节一致，12 次重复检查通过。
- 完整 Qwen 8K：所有 rank 的全量输出 hash 与旧版相同；保存的 129-token 输出和输入前缀逐字节相同。
- Qwen 与 Flash 维度的 BWD：各 160 份 tensor 与此前保存的 main 参考逐字节一致，包括 dW1/dW2；native 重复检查通过。
  此项使用 E32、每 rank 129 tokens；Flash 为 H4096/I2048/K6。
- 动态/静态 buffer、偏斜路由、保留状态以及 6 次 FWD+BWD Graph 重放通过，检查所有梯度位模式。
- 恢复后的版本：EP8/8K 在带诊断和正常设置下分别通过 128 次重复；EP8/64K 通过 32+14 次，EP16/64K 通过 24+14 次。
  输出前缀与旧版逐字节一致。正常计时与带诊断的压力检查分开记录。
- 33 个 host tests 通过，覆盖 rolling counter 回绕、乱序 slot 完成、资源配置及共享参数校验。

## 未保留的尝试与诊断边界

增加 Return streams 到 8（128 行包）曾将 EP16 Return 拉长到约 4.8 ms；降低到 2 后恢复到约 3.17 ms。
增加 sender CTA、按 channel 分配 data context、循环错开 rank 的发送起点均未解决该差距。
4 streams 配合 4 槽、保持 Return buffer 容量不变的完整 FWD 也更慢，未采用。
不能仅凭这些结果把根因归结为 context 共享、ring 显存容量或起始 incast。

独立 ready 观察点显示，接收端 ready → claim 中位数约 3–5 us；慢配置的 sender 约 2.4 ms 已提交完毕，
ready 则持续到约 5.6 ms。延迟主要位于提交之后、远端 ready 之前，而非 copy warp 长时间不领取。
两物理端口的总流量接近均分；这不能排除时序上的拥塞。

Return ready bitmap 实验虽然通过小规模 FWD/BWD、Graph 和一次 64K 检查，仍在 EP8/8K 出现 watchdog 超时。
该实验未带来明确性能收益，根因尚未确认，已撤销，不能把其通过的检查当作最终版本稳定性的替代证据。
