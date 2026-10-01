# Static V4 ring inference profile

H=2048、I=512、E=64、K=4，L20X 核心 1620 MHz。单机 EP8 从 512、双机
EP16 从 2048 tokens/rank 切换至宽 tile；均采用 packed gate8/up8。
协议不依赖双机，但三机以上未做实机测试，阈值也不是所有模型的通用最优值。

构建时选择本目录，并保持所有翻译单元使用相同的 packet 配置：

```sh
export NVCCFLAGS="-I$PWD/configs/static_inference_ring_20260924 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -DWEFTMOE_FWD_STATIC_V4_NETWORK_LANES=32 -DWEFTMOE_FWD_STATIC_V4_PACKET_ROWS=8 -DWEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS=16"
python3 build_native.py lib --jobs 8 --output build/static_v4_profiles_20260924/libweftmoe.so
```

ring 仍显式启用。该配置已接通 inference 及训练 ring Forward；训练保存
canonical checkpoints，衔接静态 Backward。不启用 ring 时保持原路径：

```sh
export WEFTMOE_STATIC_RING_TOKENS=512
export WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS=8
```

详细消融、数值验收与未完成项见
[ring 记录](../../docs/static_forward_v4_ring_record.md) 的 DC 节及后续验收。
Return Qmax16见ET：在ES完成机制上，双机8K约快1.2%、20K约快0.4%–0.5%，
pool少约16.7 MB/rank；训练Forward对照持平，单机计算代码不变。
训练对接、实际容量分配和性能限制见
[训练记录](../../docs/static_forward_v4_training_record.md) 的 M6d–M6h、FM–FS2。
FM由FC1保存SavedX，移除Pull的第二次写回；长文训练Forward有所改善，
FO将其不变目的地址移出K循环；均不改变inference计算代码。训练专用窗口
容量消融单独记录于FP，尚不引入自动切换。FR3仅单机训练采用stable-source
物理行与M32 SavedX TMA，相对FO长文main再改善约5%–6%；网络保留原布局，
完整回归持平。FS2将tape准备放到物理规划之前，移除冗余runtime回传和host
等待；单机训练长文wall改善约1%–2%，双机持平，GPU计算代码不变。
FT将完整M128的FC1 S改为TMA写回，沿用B scratch与完成链；单机20K约快
0.3%–0.5%，双机及训练持平，没有改窗口、packet或GEMM超参。
仍未达到全面领先V3的性能目标。

2026-10-01 的训练路径使用 count-only mini Notify、按实际 R/P 分配 tape，
planner 进入共享 PF main。静态 BWD 在全部本地 dX 完成后才允许 dW；
Return/Reduce 独立推进。当前 H2048/I512 profile 的 BWD math/control
寄存器配额为 208/88，sender CTA 仍 16；Forward 的配额不变。这是长文
实测配置，不作为其它模型的通用最优值。对应训练记录末尾 Y2：EP8/EP16
两档梯度 hash、长文完整 Torch/多 tape/失败重试通过，双机 8K BWD 与 V3
接近，完整训练性能门禁仍在继续。

权重原地更新前须调用 `invalidate_inference_layer` 并重建相应 graphs；
完整合同见[设计第 10 节](../../docs/static_forward_v4_design.md)。
