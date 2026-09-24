# Static V4 ring inference profile

H=2048、I=512、E=64、K=4，L20X 核心 1620 MHz。单机 EP8 从 512、双机
EP16 从 2048 tokens/rank 切换至宽 tile；均采用 packed gate8/up8。
协议不依赖双机，但三机以上未做实机测试，阈值也不是所有模型的通用最优值。

构建时选择本目录，并保持所有翻译单元使用相同的 packet 配置：

```sh
export NVCCFLAGS="-I$PWD/configs/static_inference_ring_20260924 -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -DWEFTMOE_FWD_STATIC_V4_NETWORK_LANES=32 -DWEFTMOE_FWD_STATIC_V4_PACKET_ROWS=8 -DWEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS=32"
python3 build_native.py lib --jobs 8 --output build/static_v4_profiles_20260924/libweftmoe.so
```

ring 仍显式启用，不改变训练路径：

```sh
export WEFTMOE_STATIC_RING_TOKENS=512
export WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS=8
```

详细消融、数值验收与未完成项见
[ring 记录](../../docs/static_forward_v4_ring_record.md) 的 DC 节及后续验收。
权重原地更新前须调用 `invalidate_inference_layer` 并重建相应 graphs；
完整合同见[设计第 10 节](../../docs/static_forward_v4_design.md)。
