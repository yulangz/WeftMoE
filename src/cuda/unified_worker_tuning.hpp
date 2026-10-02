#pragma once

// Phase-specific resources take precedence. Legacy WEFTMOE_UNIFIED_* inputs
// remain shared defaults, preserving frozen headers and direct compiler flags.
#ifndef WEFTMOE_UNIFIED_THREADS
#define WEFTMOE_UNIFIED_THREADS 384
#endif
#ifndef WEFTMOE_UNIFIED_COMM_THREADS
#define WEFTMOE_UNIFIED_COMM_THREADS 96
#endif
#ifndef WEFTMOE_UNIFIED_COMM_BASE
#define WEFTMOE_UNIFIED_COMM_BASE 288
#endif
#ifndef WEFTMOE_UNIFIED_MATH_REGISTERS
#define WEFTMOE_UNIFIED_MATH_REGISTERS 216
#endif
#ifndef WEFTMOE_UNIFIED_CONTROL_REGISTERS
#define WEFTMOE_UNIFIED_CONTROL_REGISTERS 64
#endif
#ifndef WEFTMOE_UNIFIED_PRODUCER_REGISTERS
#define WEFTMOE_UNIFIED_PRODUCER_REGISTERS 0
#endif
#ifndef WEFTMOE_UNIFIED_SEGMENT_BYTES
#define WEFTMOE_UNIFIED_SEGMENT_BYTES 4096
#endif
#ifndef WEFTMOE_UNIFIED_COPY_BUFFERS
// Absent an explicit shared override, prefetch uses two slots. The
// phase-specific definitions below still take priority.
#ifndef WEFTMOE_FWD_UNIFIED_COPY_BUFFERS
#define WEFTMOE_FWD_UNIFIED_COPY_BUFFERS 2
#endif
#define WEFTMOE_UNIFIED_COPY_BUFFERS 1
#endif
#ifndef WEFTMOE_UNIFIED_ROW_PREFETCH
#define WEFTMOE_UNIFIED_ROW_PREFETCH 1
#endif
#ifndef WEFTMOE_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT
#define WEFTMOE_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT 0
#endif
#ifndef WEFTMOE_UNIFIED_RETURN_DRAIN_ROWS
#define WEFTMOE_UNIFIED_RETURN_DRAIN_ROWS 128
#endif
#ifndef WEFTMOE_UNIFIED_SPLIT_NVLINK
#define WEFTMOE_UNIFIED_SPLIT_NVLINK 0
#endif
#ifndef WEFTMOE_UNIFIED_GIN_CONTEXT_PROBES
#define WEFTMOE_UNIFIED_GIN_CONTEXT_PROBES 2
#endif
#ifndef WEFTMOE_UNIFIED_GIN_PACK_SLOTS
#define WEFTMOE_UNIFIED_GIN_PACK_SLOTS 3
#endif

#ifndef WEFTMOE_FWD_UNIFIED_THREADS
#define WEFTMOE_FWD_UNIFIED_THREADS WEFTMOE_UNIFIED_THREADS
#endif
#ifndef WEFTMOE_FWD_UNIFIED_COMM_THREADS
#define WEFTMOE_FWD_UNIFIED_COMM_THREADS WEFTMOE_UNIFIED_COMM_THREADS
#endif
#ifndef WEFTMOE_FWD_UNIFIED_COMM_BASE
#define WEFTMOE_FWD_UNIFIED_COMM_BASE WEFTMOE_UNIFIED_COMM_BASE
#endif
#ifndef WEFTMOE_FWD_UNIFIED_MATH_REGISTERS
#define WEFTMOE_FWD_UNIFIED_MATH_REGISTERS WEFTMOE_UNIFIED_MATH_REGISTERS
#endif
#ifndef WEFTMOE_FWD_UNIFIED_CONTROL_REGISTERS
#define WEFTMOE_FWD_UNIFIED_CONTROL_REGISTERS WEFTMOE_UNIFIED_CONTROL_REGISTERS
#endif
#ifndef WEFTMOE_FWD_UNIFIED_PRODUCER_REGISTERS
#define WEFTMOE_FWD_UNIFIED_PRODUCER_REGISTERS WEFTMOE_UNIFIED_PRODUCER_REGISTERS
#endif
#ifndef WEFTMOE_FWD_UNIFIED_SEGMENT_BYTES
#define WEFTMOE_FWD_UNIFIED_SEGMENT_BYTES WEFTMOE_UNIFIED_SEGMENT_BYTES
#endif
#ifndef WEFTMOE_FWD_UNIFIED_COPY_BUFFERS
#define WEFTMOE_FWD_UNIFIED_COPY_BUFFERS WEFTMOE_UNIFIED_COPY_BUFFERS
#endif
#ifndef WEFTMOE_FWD_UNIFIED_ROW_PREFETCH
#define WEFTMOE_FWD_UNIFIED_ROW_PREFETCH WEFTMOE_UNIFIED_ROW_PREFETCH
#endif
#ifndef WEFTMOE_FWD_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT
#define WEFTMOE_FWD_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT WEFTMOE_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT
#endif
#ifndef WEFTMOE_FWD_UNIFIED_RETURN_DRAIN_ROWS
#define WEFTMOE_FWD_UNIFIED_RETURN_DRAIN_ROWS WEFTMOE_UNIFIED_RETURN_DRAIN_ROWS
#endif
#ifndef WEFTMOE_FWD_UNIFIED_SPLIT_NVLINK
#define WEFTMOE_FWD_UNIFIED_SPLIT_NVLINK WEFTMOE_UNIFIED_SPLIT_NVLINK
#endif
// Zero keeps the default Backward sender quota; it does not inherit Forward's.
#ifndef WEFTMOE_BWD_UNIFIED_RDMA_CTAS
#define WEFTMOE_BWD_UNIFIED_RDMA_CTAS 0
#endif
#ifndef WEFTMOE_BWD_UNIFIED_THREADS
#define WEFTMOE_BWD_UNIFIED_THREADS WEFTMOE_UNIFIED_THREADS
#endif
#ifndef WEFTMOE_BWD_UNIFIED_COMM_THREADS
#define WEFTMOE_BWD_UNIFIED_COMM_THREADS WEFTMOE_UNIFIED_COMM_THREADS
#endif
#ifndef WEFTMOE_BWD_UNIFIED_COMM_BASE
#define WEFTMOE_BWD_UNIFIED_COMM_BASE WEFTMOE_UNIFIED_COMM_BASE
#endif
#ifndef WEFTMOE_BWD_UNIFIED_MATH_REGISTERS
#define WEFTMOE_BWD_UNIFIED_MATH_REGISTERS WEFTMOE_UNIFIED_MATH_REGISTERS
#endif
#ifndef WEFTMOE_BWD_UNIFIED_CONTROL_REGISTERS
#define WEFTMOE_BWD_UNIFIED_CONTROL_REGISTERS WEFTMOE_UNIFIED_CONTROL_REGISTERS
#endif
#ifndef WEFTMOE_BWD_UNIFIED_PRODUCER_REGISTERS
#define WEFTMOE_BWD_UNIFIED_PRODUCER_REGISTERS WEFTMOE_UNIFIED_PRODUCER_REGISTERS
#endif
#ifndef WEFTMOE_BWD_UNIFIED_SEGMENT_BYTES
#define WEFTMOE_BWD_UNIFIED_SEGMENT_BYTES WEFTMOE_UNIFIED_SEGMENT_BYTES
#endif
#ifndef WEFTMOE_BWD_UNIFIED_COPY_BUFFERS
// Destination-pull keeps two independent TMA loads in flight across rows.
#define WEFTMOE_BWD_UNIFIED_COPY_BUFFERS 2
#endif
#ifndef WEFTMOE_BWD_UNIFIED_ROW_PREFETCH
#define WEFTMOE_BWD_UNIFIED_ROW_PREFETCH WEFTMOE_UNIFIED_ROW_PREFETCH
#endif
#ifndef WEFTMOE_BWD_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT
#define WEFTMOE_BWD_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT WEFTMOE_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT
#endif
#ifndef WEFTMOE_BWD_UNIFIED_RETURN_DRAIN_ROWS
#define WEFTMOE_BWD_UNIFIED_RETURN_DRAIN_ROWS WEFTMOE_UNIFIED_RETURN_DRAIN_ROWS
#endif
#ifndef WEFTMOE_BWD_UNIFIED_SPLIT_NVLINK
#define WEFTMOE_BWD_UNIFIED_SPLIT_NVLINK WEFTMOE_UNIFIED_SPLIT_NVLINK
#endif

namespace weftmoe::cuda::unified_worker_tuning {
template <unsigned Threads, unsigned CommThreads, unsigned CommBase,
          unsigned MathRegisters, unsigned ControlRegisters,
          unsigned ProducerRegisters, unsigned SegmentBytes,
          unsigned CopyBuffers>
struct Resources {
  static constexpr unsigned kUnifiedThreads = Threads;
  static constexpr unsigned kUnifiedCommThreads = CommThreads;
  static constexpr unsigned kUnifiedCommBase = CommBase;
  static constexpr unsigned kUnifiedMathRegisters = MathRegisters;
  static constexpr unsigned kUnifiedControlRegisters = ControlRegisters;
  static constexpr unsigned kUnifiedProducerRegisters =
      ProducerRegisters ? ProducerRegisters : ControlRegisters;
  static constexpr unsigned kUnifiedSegmentBytes = SegmentBytes;
  static constexpr unsigned kUnifiedCopyBuffers = CopyBuffers;
  static constexpr unsigned kUnifiedInitialRegisters =
      (65536U / Threads / 8U) * 8U;
  static constexpr unsigned kUnifiedRegisterPool = kUnifiedInitialRegisters * Threads;
  static_assert(Threads >= 384U && Threads <= 1024U && Threads % 128U == 0U);
  static_assert(CommBase >= 288U && CommBase % 32U == 0U &&
                CommThreads >= 64U && CommThreads % 32U == 0U,
                "Unified communication must occupy complete warps after the producer");
  static_assert(CommBase + CommThreads <= Threads);
  static_assert(CommBase >= 384U || kUnifiedProducerRegisters == ControlRegisters,
                "A shared producer/communication warpgroup needs one quota");
  static_assert(MathRegisters >= 24U && MathRegisters <= 256U &&
                ControlRegisters >= 24U && ControlRegisters <= 256U &&
                kUnifiedProducerRegisters >= 24U && kUnifiedProducerRegisters <= 256U &&
                MathRegisters % 8U == 0U && ControlRegisters % 8U == 0U &&
                kUnifiedProducerRegisters % 8U == 0U);
  // PTX .inc/.dec must respect the initial per-thread maximum, as well as
  // the CTA pool rounded to Hopper's allocation granularity.
  static_assert(MathRegisters >= kUnifiedInitialRegisters,
                "Math setmaxnreg.inc cannot reduce the initial register maximum");
  static_assert(ControlRegisters <= kUnifiedInitialRegisters &&
                kUnifiedProducerRegisters <= kUnifiedInitialRegisters,
                "Control setmaxnreg.dec cannot increase the initial register maximum");
  static_assert(256U * MathRegisters + 128U * kUnifiedProducerRegisters +
                    (Threads - 384U) * ControlRegisters <= kUnifiedRegisterPool);
  static_assert(SegmentBytes >= 16U && SegmentBytes % 16U == 0U);
  static_assert(CopyBuffers == 1U || CopyBuffers == 2U);
};
} // namespace weftmoe::cuda::unified_worker_tuning
