// Application tensor bindings. Python owns tensors; native handles retain their owners.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime_api.h>
#include <weftmoe/cuda/backward.hpp>
#include <cstdlib>
#include <map>
#include <optional>

namespace py = pybind11;
using namespace weftmoe;
using namespace weftmoe::cuda;

namespace {
std::shared_ptr<const void> owner(const py::object &value) {
  return {new py::object(value), [](const void *p) {
    py::gil_scoped_acquire gil;
    delete static_cast<const py::object *>(p);
  }};
}

void *pointer(const py::object &tensor) {
  return reinterpret_cast<void *>(tensor.attr("data_ptr")().cast<std::uintptr_t>());
}

std::uint64_t elements(const py::object &tensor) {
  return tensor.attr("numel")().cast<std::uint64_t>();
}

CudaTensorBuffer binding(const py::object &tensor) {
  return {pointer(tensor), elements(tensor) * tensor.attr("element_size")().cast<unsigned>(),
          owner(tensor)};
}

struct Tape {
  DeviceDispatchHandle dispatch;
  DeviceForwardHandle forward;
  NcclSession generation;
  py::object output;
};

void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct InferenceBucket {
  DeviceNotifyResourcePool notify_pool;
  NcclSession generation;
  DeviceDispatchHandle dispatch;
  py::object routes, gates;
};

struct InferenceWeights {
  ForwardWeightLease lease;
  std::uintptr_t w1, w2;
};

struct InferencePlan {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  DeviceForwardHandle forward;
  py::object output;
  py::object x, routes, gates, w1_owner, w2_owner;
  std::uintptr_t x_address = 0, routes_address = 0, gates_address = 0;
  std::uintptr_t w1 = 0, w2 = 0;
  ~InferencePlan() {
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (graph) (void)cudaGraphDestroy(graph);
  }
};

class Runtime {
 public:
  Runtime(py::bytes bootstrap, unsigned rank, unsigned world, unsigned nodes,
          unsigned device, std::uintptr_t stream, py::object stream_owner,
          unsigned experts, unsigned top_k, unsigned hidden, unsigned intermediate,
          std::uint64_t max_tokens, std::uint64_t max_rows, bool training,
          unsigned max_saved, py::dict tuning = py::dict())
      : rank_(rank), hidden_(hidden), intermediate_(intermediate),
        max_tokens_(max_tokens), training_(training), stream_(reinterpret_cast<cudaStream_t>(stream)) {
    if (!world || !nodes || world % nodes || experts % world || !top_k || !max_saved)
      throw std::invalid_argument("invalid EP topology or saved-tensor capacity");
    const auto *ring_setting = std::getenv("WEFTMOE_STATIC_RING_TOKENS");
    const auto ring_tokens = tuning.contains("window_tokens")
        ? tuning["window_tokens"].cast<unsigned>()
        : ring_setting ? std::stoul(ring_setting) : 0U;
    const bool training_ring = training && ring_tokens;
    const auto peer_capacity_rows = max_rows;
    if (training_ring) {
      // Only canonical routing/scheduler metadata uses this worst-case
      // capacity. X/S stay bounded and saved tensors use R; Backward peer
      // storage starts with the caller's capacity hint, not the global maximum.
      if (max_tokens > UINT32_MAX / top_k / world)
        throw std::invalid_argument("training ring metadata exceeds uint32 row indices");
      max_rows = std::max(max_rows, max_tokens * top_k * world);
    }
    std::string bytes = bootstrap;
    auto id = NcclBootstrapId::from_bytes({bytes.begin(), bytes.end()});
    execution_ = make_external_cuda_execution("torch", device, stream_, owner(stream_owner));
    base_ = make_nccl_session("weftmoe-torch", world, rank, 1, device, id);
    config_.num_experts = experts;
    config_.top_k = top_k;
    config_.m_tile_rows = 64;
    config_.topology.world_size = world;
    config_.topology.num_nodes = nodes;
    for (unsigned r = 0; r < world; ++r) {
      config_.topology.rank_to_node.push_back(r / (world / nodes));
      config_.topology.rank_to_local_rank.push_back(r % (world / nodes));
    }
    for (unsigned e = 0; e < experts; ++e)
      config_.topology.expert_to_rank.push_back(e / (experts / world));
    config_.capacity.workspace_rows_per_rank.assign(world, max_rows);
    fused_inference_ =
        !training &&
        std::getenv("WEFTMOE_DISABLE_SINGLE_NODE_INFERENCE") == nullptr;
    notify_pool_ = make_device_notify_resource_pool(
        base_, execution_,
        {config_, max_tokens, 8192, max_saved, 4,
         fused_inference_, training_ring});
    if (!WEFTMOE_ENABLE_GIN_RDMA && nodes != 1)
      throw std::invalid_argument(
          "this build sets WEFTMOE_ENABLE_GIN_RDMA=0 and supports a "
          "single-node EP team only");
    ActivationTransportPoolConfig capacity;
    capacity.max_tokens_per_rank = max_tokens;
    capacity.max_hidden_size = hidden;
    capacity.max_intermediate_size = intermediate;
    capacity.max_top_k = top_k;
    capacity.max_world_size = world;
    capacity.max_nodes = nodes;
    capacity.max_owned_experts = experts / world;
    capacity.max_destination_rows_per_rank = max_rows;
    capacity.backward_peer_capacity_rows = peer_capacity_rows;
    capacity.max_inflight_generations = 1;
    capacity.channels = 1;
    capacity.ring_slots_per_source_channel = 1;
    capacity.enable_backward_state = training;
    capacity.static_forward_window_tokens = ring_tokens;
    if (const auto *depth = std::getenv("WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS"))
      capacity.static_forward_contribution_windows = std::stoul(depth);
    if (tuning.contains("contribution_windows"))
      capacity.static_forward_contribution_windows =
          tuning["contribution_windows"].cast<unsigned>();
    capacity.enable_gin_rdma =
        WEFTMOE_ENABLE_GIN_RDMA && nodes > 1;
    pool_ = prepare_activation_transport_pool(base_.session(), execution_, capacity);
#if WEFTMOE_ENABLE_GIN_RDMA
    if (WEFTMOE_ENABLE_GIN_RDMA && nodes > 1)
      gin_ = std::make_unique<GinRdmaForwardMegaKernel>(base_, execution_, pool_);
    else
      local_ = std::make_unique<LocalCudaP2pForwardMegaKernel>(base_, execution_, pool_);
#else
    local_ = std::make_unique<LocalCudaP2pForwardMegaKernel>(base_, execution_, pool_);
#endif
    torch_ = py::module_::import("torch");
    device_ = "cuda:" + std::to_string(device);
    if (training_)
      tensor_preparation_ =
          std::make_unique<CollectiveTensorPreparation>(base_, execution_);
    check_cuda(cudaEventCreateWithFlags(&input_ready_, cudaEventDisableTiming));
  }

  ~Runtime() {
    // eager_inference_async() deliberately keeps the most recent handle alive
    // after returning its output.  Drain it while the backend/pools and Python
    // tensor owners are still valid.  Destructors cannot report an error, so
    // explicit users should call drain_eager_inference() at a phase boundary.
    try { drain_eager_inference(); } catch (...) {}
    if (input_ready_) (void)cudaEventDestroy(input_ready_);
  }

  py::object eager_inference(py::object x, py::object routes, py::object gates,
                             py::object w1, py::object w2, std::uintptr_t caller_stream,
                             std::optional<unsigned> layer = std::nullopt) {
    if (training_) throw std::invalid_argument("eager_inference is inference-only");
    drain_eager_inference();
    auto caller = reinterpret_cast<cudaStream_t>(caller_stream);
    check_cuda(cudaEventRecord(input_ready_, caller));
    check_cuda(cudaStreamWaitEvent(stream_, input_ready_));
    return execute_forward(x, routes, gates, w1, w2, true, layer)->output;
  }

  py::object eager_inference_async(py::object x, py::object routes,
                                   py::object gates, py::object w1,
                                   py::object w2,
                                   std::uintptr_t caller_stream,
                                   std::optional<unsigned> layer = std::nullopt) {
    if (training_)
      throw std::invalid_argument("eager_inference_async is inference-only");
    // There is one persistent transport workspace.  Reclaim the preceding
    // generation at the next MoE boundary, after Python has had a chance to
    // enqueue all intervening layer work on the caller stream.
    drain_eager_inference();
    auto caller = reinterpret_cast<cudaStream_t>(caller_stream);
    check_cuda(cudaEventRecord(input_ready_, caller));
    check_cuda(cudaStreamWaitEvent(stream_, input_ready_));
    auto tape = execute_forward(x, routes, gates, w1, w2, false, layer);
    // Preserve ordinary PyTorch stream semantics without completing the work
    // on the host: consumers on caller wait for the output-ready CUDA event.
    tape->forward.wait_workspace_on(caller);
    auto output = tape->output;
    pending_inference_ = std::move(tape);
    return output;
  }

  py::dict profile_inference(py::object x, py::object routes,
                             py::object gates, py::object w1, py::object w2,
                             std::uintptr_t caller_stream,
                             std::optional<unsigned> layer = std::nullopt) {
    if (training_)
      throw std::invalid_argument("profile_inference is inference-only");
    drain_eager_inference();
    auto caller = reinterpret_cast<cudaStream_t>(caller_stream);
    check_cuda(cudaEventRecord(input_ready_, caller));
    check_cuda(cudaStreamWaitEvent(stream_, input_ready_));
    auto tape = execute_forward(x, routes, gates, w1, w2, true, layer);
    const auto &timings = tape->forward.timings();
    py::dict result;
    result["output"] = tape->output;
    result["total_us"] = timings.total_us;
    result["route_exchange_us"] = timings.route_exchange_us;
    result["dispatch_us"] = timings.dispatch_us;
    result["fc1_swiglu_us"] = timings.fc1_swiglu_us;
    result["fc2_us"] = timings.fc2_us;
    result["return_us"] = timings.combine_return_us;
    result["reduce_us"] = timings.reduce_us;
    return result;
  }

  void drain_eager_inference() {
    if (!pending_inference_)
      return;
    auto pending = std::move(pending_inference_);
    pending->forward.wait();
  }

  std::shared_ptr<Tape> forward(py::object x, py::object routes, py::object gates,
                                py::object w1, py::object w2) {
    return execute_forward(x, routes, gates, w1, w2, true);
  }

  std::shared_ptr<Tape> execute_forward(py::object x, py::object routes,
                                        py::object gates, py::object w1,
                                        py::object w2, bool synchronize,
                                        std::optional<unsigned> layer = std::nullopt) {
    cudaStreamCaptureStatus status;
    if (cudaStreamIsCapturing(stream_, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone)
      throw std::runtime_error("dynamic forward requires preparation before CUDA Graph capture");
    auto tape = std::make_shared<Tape>();
    auto tokens = elements(x) / hidden_;
    if (tokens > max_tokens_)
      throw std::invalid_argument("token capacity exceeded");
    tape->generation = make_nccl_generation(base_, "weftmoe-torch", ++generation_);
    auto context = make_rank_collective_context(tape->generation.session(), rank_, execution_.token());
    auto producer = make_external_cuda_producer(context, execution_.device_ordinal(), stream_,
                                                owner(x), "torch");
    auto r = producer.publish_selected_experts(static_cast<const std::int32_t *>(pointer(routes)),
                                               elements(routes), tokens, config_.top_k, owner(routes));
    auto g = producer.publish_gates(static_cast<const float *>(pointer(gates)), elements(gates), r, owner(gates));
    r = import_selected_expert_routes(r, context, execution_);
    g = import_selected_expert_gates(g, r, context, execution_);
    CudaNotifyGenerationCoordinator coordinator(tape->generation, execution_, notify_pool_);
    tape->dispatch = coordinator.run_device({config_, r, g, max_tokens_}).dispatch();
    BFloat16ActivationLease input;
    ForwardWeightLease weights;
    ForwardMegaKernelConfig fc{hidden_, intermediate_, training_};
    prepare_tensors([&] {
      input = producer.publish_activations(
          static_cast<const transport::BFloat16 *>(pointer(x)),
          elements(x), tokens, hidden_, owner(x));
      if (layer) {
        weights = prepared_inference_weights(*layer, producer, w1, w2);
      } else {
        auto packed_w1 = empty({elements(w1)}, "bfloat16");
        weights = producer.publish_weights(
            static_cast<const transport::BFloat16 *>(pointer(w1)), elements(w1), owner(w1),
            static_cast<const transport::BFloat16 *>(pointer(w2)), elements(w2), owner(w2),
            config_.num_experts / config_.topology.world_size,
            rank_ * (config_.num_experts / config_.topology.world_size), hidden_, intermediate_,
            binding(packed_w1));
      }
      // Mini Notify already supplies the exact tape size. Prepare inputs and
      // saved tensors under one allocation agreement before entering main.
      tape->output = empty({tokens, hidden_}, "bfloat16");
      ForwardTensorBuffers buffers;
      buffers.output = binding(tape->output);
      fc.tensors = prepare_forward_tensors(tape->dispatch, execution_, fc,
                                           buffers, allocator());
    });
#if WEFTMOE_ENABLE_GIN_RDMA
    if (synchronize)
      tape->forward = gin_ ? gin_->run(tape->dispatch, input, weights, fc)
                           : local_->run(tape->dispatch, input, weights, fc);
    else
      tape->forward = gin_ ? gin_->enqueue(tape->dispatch, input, weights, fc)
                           : local_->enqueue(tape->dispatch, input, weights, fc);
#else
    tape->forward = synchronize
                        ? local_->run(tape->dispatch, input, weights, fc)
                        : local_->enqueue(tape->dispatch, input, weights, fc);
#endif
    return tape;
  }

  py::tuple backward(std::shared_ptr<Tape> tape, py::object dy) {
    if (!training_ || !tape->forward)
      throw std::invalid_argument("backward requires a live training tape");
    auto context = make_rank_collective_context(tape->generation.session(), rank_, execution_.token());
    auto producer = make_external_cuda_producer(context, execution_.device_ordinal(), stream_, owner(dy), "torch-dy");
    auto tokens = elements(dy) / hidden_;
    auto input = producer.publish_output_gradients(static_cast<const transport::BFloat16 *>(pointer(dy)),
                                                   elements(dy), tokens, hidden_, owner(dy));
    auto local_experts = config_.num_experts / config_.topology.world_size;
    py::object dx, dg, dw1, dw2;
    BackwardMegaKernelConfig bc{hidden_, intermediate_};
    prepare_tensors([&] {
      dx = empty({tokens, hidden_}, "bfloat16");
      dg = empty({tokens, config_.top_k}, "float32");
      dw1 = empty({local_experts, 2 * intermediate_, hidden_}, "bfloat16");
      dw2 = empty({local_experts, hidden_, intermediate_}, "bfloat16");
      bc.tensors = BackwardMegaKernel().prepare_tensors(tape->forward, bc,
          {binding(dx), binding(dg), binding(dw1), binding(dw2)}, allocator());
    });
    BackwardMegaKernel().run(tape->forward, input, bc);
    tape->forward = {};
    tape->dispatch = {};
    return py::make_tuple(dx, dg, dw1, dw2);
  }

  py::dict usage() const {
    auto p = pool_.usage();
    auto n = notify_pool_.usage();
    py::dict result;
    result["compute_bytes"] = p.compute_device_bytes;
    result["gin_bytes"] = p.gin_registered_bytes;
    result["notify_bytes"] = n.device_bytes;
    result["notify_slots_in_use"] = n.slots_in_use;
    return result;
  }

  void retire_peer_mappings() {
    // TensorRuntime.close() has synchronized all ranks; outer application
    // graphs must already be discarded. Drop internal plans before unmapping
    // any of their addresses, but keep exports alive until the next barrier.
    plans_.clear();
    buckets_.clear();
    inference_weights_.clear();
    pool_.retire_peer_mappings();
  }

  py::object graph_forward(unsigned layer, py::object x, py::object routes,
                           py::object gates, py::object w1, py::object w2,
                           std::uintptr_t caller_stream) {
    if (training_) throw std::invalid_argument("graph_forward is inference-only");
    auto caller = reinterpret_cast<cudaStream_t>(caller_stream);
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(caller, &capture));
    if (pending_inference_) {
      if (capture != cudaStreamCaptureStatusNone)
        throw std::runtime_error(
            "drain eager inference before CUDA Graph capture");
      drain_eager_inference();
    }
    const auto tokens = elements(x) / hidden_;
    const auto key = std::make_pair(layer, tokens);
    if (plans_.find(key) == plans_.end()) {
      if (capture != cudaStreamCaptureStatusNone)
        throw std::runtime_error("Warm every inference layer/token bucket before outer graph capture");
      // All graph preparation is outside the application's capture and hot path.
      check_cuda(cudaDeviceSynchronize());
      if (buckets_.find(tokens) == buckets_.end()) prepare_bucket(tokens);
      prepare_plan(key, x, routes, gates, w1, w2);
    }
    auto &plan = *plans_.at(key);
    if (plan.x_address != reinterpret_cast<std::uintptr_t>(pointer(x)) ||
        plan.routes_address !=
            reinterpret_cast<std::uintptr_t>(pointer(routes)) ||
        plan.gates_address !=
            reinterpret_cast<std::uintptr_t>(pointer(gates)))
      throw std::runtime_error(
          "Inference input addresses changed: use stable CUDA Graph buffers "
          "or call invalidate_inference_layer(layer) before re-preparation");
    if (plan.w1 != reinterpret_cast<std::uintptr_t>(pointer(w1)) ||
        plan.w2 != reinterpret_cast<std::uintptr_t>(pointer(w2)))
      throw std::runtime_error(
          "Expert addresses changed: discard outer graphs and call "
          "invalidate_inference_layer(layer) before re-preparation");
    if (capture == cudaStreamCaptureStatusNone) {
      check_cuda(cudaGraphLaunch(plan.executable, caller));
    } else {
      cudaGraph_t parent;
      const cudaGraphNode_t *dependencies = nullptr;
      std::size_t count = 0;
      unsigned long long capture_id = 0;
      check_cuda(cudaStreamGetCaptureInfo_v2(caller, &capture, &capture_id, &parent,
                                             &dependencies, &count));
      cudaGraphNode_t child;
      check_cuda(cudaGraphAddChildGraphNode(&child, parent, dependencies, count,
                                            plan.graph));
      check_cuda(cudaStreamUpdateCaptureDependencies(caller, &child, 1,
                                                     cudaStreamSetCaptureDependencies));
    }
    return plan.output;
  }

  void invalidate_inference_layer(unsigned layer, std::uintptr_t caller_stream) {
    if (training_)
      throw std::invalid_argument("inference plans are unavailable in training mode");
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(caller_stream),
                                     &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw std::runtime_error("invalidate inference plans outside CUDA Graph capture");
    drain_eager_inference();
    // Explicit model-update boundary, never a replay-path version check. The
    // caller must first discard outer graphs that reference this layer's plans.
    check_cuda(cudaDeviceSynchronize());
    auto first = plans_.lower_bound({layer, 0U});
    auto last = first;
    while (last != plans_.end() && last->first.first == layer)
      ++last;
    plans_.erase(first, last);
    inference_weights_.erase(layer);
  }

 private:
  const ForwardWeightLease &prepared_inference_weights(
      unsigned layer, ExternalCudaProducer &producer,
      const py::object &w1, const py::object &w2) {
    const auto w1_address = reinterpret_cast<std::uintptr_t>(pointer(w1));
    const auto w2_address = reinterpret_cast<std::uintptr_t>(pointer(w2));
    auto entry = inference_weights_.find(layer);
    if (entry == inference_weights_.end()) {
      // The immutable packed copy belongs to a resident layer, not a token
      // bucket or invocation. Eager and graph inference share its lifetime.
      auto weight_producer = make_external_cuda_producer(
          producer.context(), execution_.device_ordinal(), stream_,
          owner(w1), "inference-weights");
      auto weights = weight_producer.publish_weights(
          static_cast<const transport::BFloat16 *>(pointer(w1)), elements(w1), owner(w1),
          static_cast<const transport::BFloat16 *>(pointer(w2)), elements(w2), owner(w2),
          config_.num_experts / config_.topology.world_size,
          rank_ * (config_.num_experts / config_.topology.world_size), hidden_, intermediate_);
      entry = inference_weights_.emplace(
          layer, InferenceWeights{std::move(weights), w1_address, w2_address}).first;
    } else if (entry->second.w1 != w1_address || entry->second.w2 != w2_address) {
      throw std::runtime_error(
          "Expert addresses changed: discard outer graphs and call "
          "invalidate_inference_layer(layer) before re-preparation");
    }
    return entry->second.lease;
  }

  void prepare_bucket(std::uint64_t tokens) {
    if (tokens > max_tokens_) throw std::invalid_argument("inference token capacity exceeded");
    auto bucket = std::make_unique<InferenceBucket>();
    bucket->routes = empty({tokens, config_.top_k}, "int32");
    auto ids = torch_.attr("arange")(config_.top_k, py::arg("device")=device_,
                                    py::arg("dtype")=torch_.attr("int32"));
    bucket->routes.attr("copy_")(ids);
    bucket->gates = empty({tokens, config_.top_k}, "float32");
    bucket->gates.attr("fill_")(1.0 / config_.top_k);
    check_cuda(cudaDeviceSynchronize());
    bucket->generation = make_nccl_generation(base_, "weftmoe-torch", ++generation_);
    auto context = make_rank_collective_context(bucket->generation.session(), rank_, execution_.token());
    auto producer = make_external_cuda_producer(context, execution_.device_ordinal(), stream_,
                                                owner(bucket->routes), "inference-bucket");
    auto routes = producer.publish_selected_experts(
        static_cast<const std::int32_t *>(pointer(bucket->routes)), elements(bucket->routes),
        tokens, config_.top_k, owner(bucket->routes));
    auto gates = producer.publish_gates(static_cast<const float *>(pointer(bucket->gates)),
                                        elements(bucket->gates), routes, owner(bucket->gates));
    routes = import_selected_expert_routes(routes, context, execution_);
    gates = import_selected_expert_gates(gates, routes, context, execution_);
    bucket->notify_pool = make_device_notify_resource_pool(
        base_, execution_,
        {config_, tokens, 8192, 1, 4, fused_inference_});
    CudaNotifyGenerationCoordinator coordinator(bucket->generation, execution_, bucket->notify_pool);
    bucket->dispatch = coordinator.run_device({config_, routes, gates, tokens}).dispatch();
    buckets_[tokens] = std::move(bucket);
  }

  void prepare_plan(std::pair<unsigned, std::uint64_t> key, py::object x,
                    py::object routes, py::object gates, py::object w1,
                    py::object w2) {
    auto &bucket = *buckets_.at(key.second);
    auto context = make_rank_collective_context(bucket.generation.session(), rank_, execution_.token());
    auto producer = make_external_cuda_producer(context, execution_.device_ordinal(), stream_,
                                                owner(x), "inference-plan");
    auto activation = producer.publish_activations(
        static_cast<const transport::BFloat16 *>(pointer(x)), elements(x),
        key.second, hidden_, owner(x));
    const auto w1_address = reinterpret_cast<std::uintptr_t>(pointer(w1));
    const auto w2_address = reinterpret_cast<std::uintptr_t>(pointer(w2));
    const auto &weights = prepared_inference_weights(key.first, producer, w1, w2);
    auto plan = std::make_unique<InferencePlan>();
    plan->output = empty({key.second, hidden_}, "bfloat16");
    plan->x = x;
    plan->routes = routes;
    plan->gates = gates;
    plan->w1_owner = w1;
    plan->w2_owner = w2;
    plan->x_address = reinterpret_cast<std::uintptr_t>(pointer(x));
    plan->routes_address = reinterpret_cast<std::uintptr_t>(pointer(routes));
    plan->gates_address = reinterpret_cast<std::uintptr_t>(pointer(gates));
    plan->w1 = reinterpret_cast<std::uintptr_t>(pointer(w1));
    plan->w2 = reinterpret_cast<std::uintptr_t>(pointer(w2));
    ForwardMegaKernelConfig fc{hidden_, intermediate_, false};
    fc.serialized_graph_replay = true;
    fc.inference_routes = binding(routes);
    fc.inference_gates = binding(gates);
    ForwardTensorBuffers buffers;
    buffers.output = binding(plan->output);
    fc.tensors = prepare_forward_tensors(bucket.dispatch, execution_, fc, buffers, allocator());
    check_cuda(cudaStreamSynchronize(stream_));
    check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    bucket.dispatch.enqueue_notify(bucket.generation);
#if WEFTMOE_ENABLE_GIN_RDMA
    plan->forward = gin_ ? gin_->enqueue(bucket.dispatch, activation, weights, fc)
                         : local_->enqueue(bucket.dispatch, activation, weights, fc);
#else
#if WEFTMOE_NOTIFY_TIMING_MODE >= 1
    // Measurement-only: capture the notify stage without the forward kernel.
    plan->forward = {};
#else
    plan->forward = local_->enqueue(bucket.dispatch, activation, weights, fc);
#endif
#endif
    check_cuda(cudaStreamEndCapture(stream_, &plan->graph));
    check_cuda(cudaGraphInstantiate(&plan->executable, plan->graph, 0));
    check_cuda(cudaGraphLaunch(plan->executable, stream_));
    check_cuda(cudaStreamSynchronize(stream_));
    plan->forward.release_serialized_graph_workspace();
    plans_[key] = std::move(plan);
  }

  py::object empty(std::vector<std::uint64_t> shape, const char *dtype) {
    return torch_.attr("empty")(shape, py::arg("device") = device_, py::arg("dtype") = torch_.attr(dtype));
  }
  CudaTensorAllocator allocator() {
    return [this](std::uint64_t bytes) { return binding(empty({bytes}, "uint8")); };
  }
  void prepare_tensors(const std::function<void()> &prepare) {
    if (tensor_preparation_)
      tensor_preparation_->run(prepare);
    else
      prepare();
  }
  unsigned rank_, hidden_, intermediate_;
  std::uint64_t max_tokens_, generation_ = 1;
  bool training_, fused_inference_ = false;
  cudaStream_t stream_;
  cudaEvent_t input_ready_ = nullptr;
  RuntimeConfig config_;
  CudaExecution execution_;
  NcclSession base_;
  DeviceNotifyResourcePool notify_pool_;
  ActivationTransportResourcePool pool_;
  std::unique_ptr<CollectiveTensorPreparation> tensor_preparation_;
  std::unique_ptr<LocalCudaP2pForwardMegaKernel> local_;
#if WEFTMOE_ENABLE_GIN_RDMA
  std::unique_ptr<GinRdmaForwardMegaKernel> gin_;
#endif
  py::object torch_;
  std::string device_;
  // Destroy graph executables before their metadata, packed weights and workspace.
  std::map<unsigned, InferenceWeights> inference_weights_;
  std::map<std::uint64_t, std::unique_ptr<InferenceBucket>> buckets_;
  std::map<std::pair<unsigned, std::uint64_t>, std::unique_ptr<InferencePlan>> plans_;
  // One-generation pipeline for eager inference.  The pool intentionally has
  // one large workspace; completion is delayed to the next MoE boundary.
  std::shared_ptr<Tape> pending_inference_;
};
}

PYBIND11_MODULE(_runtime, m) {
  m.def("forward_tuning_id", &forward_tuning_id);
  m.def("bootstrap_id", [] {
    auto id = make_nccl_bootstrap_id();
    return py::bytes(reinterpret_cast<const char *>(id.bytes().data()), id.bytes().size());
  });
  py::class_<Tape, std::shared_ptr<Tape>>(m, "Tape").def_readonly("output", &Tape::output);
  py::class_<Runtime, std::shared_ptr<Runtime>>(m, "Runtime")
      .def(py::init<py::bytes, unsigned, unsigned, unsigned, unsigned, std::uintptr_t, py::object,
                    unsigned, unsigned, unsigned, unsigned, std::uint64_t, std::uint64_t, bool, unsigned>())
      .def(py::init<py::bytes, unsigned, unsigned, unsigned, unsigned, std::uintptr_t, py::object,
                    unsigned, unsigned, unsigned, unsigned, std::uint64_t, std::uint64_t, bool, unsigned, py::dict>())
      .def("forward", &Runtime::forward)
      .def("backward", &Runtime::backward)
      .def("graph_forward", &Runtime::graph_forward)
      .def("invalidate_inference_layer", &Runtime::invalidate_inference_layer)
      .def("eager_inference", &Runtime::eager_inference,
           py::arg("x"), py::arg("routes"), py::arg("gates"), py::arg("w1"), py::arg("w2"),
           py::arg("caller_stream"), py::arg("layer") = std::nullopt)
      .def("eager_inference_async", &Runtime::eager_inference_async,
           py::arg("x"), py::arg("routes"), py::arg("gates"), py::arg("w1"), py::arg("w2"),
           py::arg("caller_stream"), py::arg("layer") = std::nullopt)
      .def("profile_inference", &Runtime::profile_inference,
           py::arg("x"), py::arg("routes"), py::arg("gates"), py::arg("w1"), py::arg("w2"),
           py::arg("caller_stream"), py::arg("layer") = std::nullopt)
      .def("drain_eager_inference", &Runtime::drain_eager_inference)
      .def("retire_peer_mappings", &Runtime::retire_peer_mappings)
      .def("usage", &Runtime::usage);
}
