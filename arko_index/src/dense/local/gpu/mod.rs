//! Local models on a GPU (features `local-embed-gpu` and `local-embed-cuda`), on burn's
//! tensors: CUDA through cubecl, which compiles its kernels with NVRTC at run time and
//! loads the driver and NVRTC when it starts, or wgpu (Vulkan, Metal, DirectX 12). The
//! same encoder runs on burn's ndarray backend in the offline tests.
//!
//! A request's texts are sorted by token count, longest first, and split into padded
//! batches of at most the embedder's batch tokens (`rows × padded length`, one text
//! alone may exceed it), so padding stays small. Every batch is queued before the
//! first is read back, and the vectors return in input order.
//!
//! Every tensor stays within the device's largest buffer ([`Choice::max_buffer`]): a
//! batch holds at most as many tokens as one hidden state of that size, and the encoder
//! splits its widest work (the feed-forward layer, the query, key and value projection
//! and attention's scores) into pieces that fit. cubecl panics on its device thread
//! when an allocation is larger than that, and the device is unusable from then on.
mod encoder;

use super::{
    Device, Precision,
    model::{Config, Pooling},
    safetensors::Tensors,
};
use crate::bench_v1::Error;
use burn_tensor::{DType, FloatDType, backend::Backend};
use encoder::Encoder;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Texts are padded to a multiple of this many tokens.
const PAD: usize = 8;

/// A device a model can run on, found by [`choose`].
#[derive(Clone)]
pub(crate) enum Choice {
    #[cfg(feature = "local-embed-cuda")]
    Cuda {
        name: String,
        /// A quarter of the GPU's memory: cubecl's largest allocation on CUDA.
        max_buffer: u64,
    },
    #[cfg(feature = "local-embed-gpu")]
    Wgpu {
        device: burn_wgpu::WgpuDevice,
        name: String,
        /// The adapter's largest storage buffer, cubecl's largest allocation on wgpu
        /// (128 MiB on Mesa's llvmpipe).
        max_buffer: u64,
    },
    /// burn's ndarray backend, for tests. It allocates any size, and `max_buffer`
    /// makes the encoder split its work as a device with that limit would.
    #[cfg(test)]
    NdArray { max_buffer: u64 },
}

impl Choice {
    /// The device, for messages: `CUDA: NAME` or `wgpu: NAME (API, type)`.
    pub(crate) fn name(&self) -> String {
        match self {
            #[cfg(feature = "local-embed-cuda")]
            Choice::Cuda { name, .. } => format!("CUDA: {name}"),
            #[cfg(feature = "local-embed-gpu")]
            Choice::Wgpu { name, .. } => format!("wgpu: {name}"),
            #[cfg(test)]
            Choice::NdArray { .. } => "burn ndarray (CPU)".into(),
        }
    }

    /// The largest buffer, in bytes, the device allocates: the bound of every tensor.
    pub(crate) fn max_buffer(&self) -> u64 {
        match self {
            #[cfg(feature = "local-embed-cuda")]
            Choice::Cuda { max_buffer, .. } => *max_buffer,
            #[cfg(feature = "local-embed-gpu")]
            Choice::Wgpu { max_buffer, .. } => *max_buffer,
            #[cfg(test)]
            Choice::NdArray { max_buffer } => *max_buffer,
        }
    }
}

/// The GPU `device` stands for with `precision`: `Ok(None)` when [`Device::Auto`]
/// finds none (the CPU runs it). Errors say why a named device can't.
pub(crate) fn choose(device: Device, precision: Precision) -> Result<Option<Choice>, Error> {
    let auto_f32 = device == Device::Auto && precision == Precision::F32;
    // The adapters wgpu may take: discrete GPUs only for Auto with f32 (where the CPU
    // is the alternative), any GPU for Gpu and Auto, anything when wgpu is named.
    let adapters = match device {
        _ if auto_f32 => Adapters::Discrete,
        Device::Wgpu => Adapters::Any,
        _ => Adapters::Gpus,
    };
    let mut why = Vec::new();
    if matches!(device, Device::Auto | Device::Gpu | Device::Cuda) {
        match cuda() {
            Ok(choice) => return Ok(Some(choice)),
            Err(e) => why.push(e),
        }
    }
    if matches!(device, Device::Auto | Device::Gpu | Device::Wgpu) {
        if precision == Precision::Bf16 {
            why.push("wgpu: no bf16 arithmetic".into());
        } else {
            match wgpu(adapters) {
                Ok(choice) => return Ok(Some(choice)),
                Err(e) => why.push(e),
            }
        }
    }
    if auto_f32 {
        return Ok(None);
    }
    Err(format!("no GPU to run {precision} on: {}", why.join("; ")).into())
}

#[cfg(feature = "local-embed-cuda")]
fn cuda() -> Result<Choice, String> {
    use cudarc::driver::{result, sys};
    // SAFETY: loads the driver library, as every cudarc call does first.
    if !unsafe { sys::is_culib_present() } {
        return Err("CUDA: no NVIDIA driver (libcuda) found".into());
    }
    result::init().map_err(|e| format!("CUDA: the driver failed to start: {e}"))?;
    let count = result::device::get_count().map_err(|e| format!("CUDA: {e}"))?;
    if count < 1 {
        return Err("CUDA: the driver lists no GPU".into());
    }
    let dev = result::device::get(0).map_err(|e| format!("CUDA: {e}"))?;
    let name = result::device::get_name(dev).map_err(|e| format!("CUDA: {e}"))?;
    // SAFETY: `dev` is a device the driver listed.
    let memory = unsafe { result::device::total_mem(dev) }.map_err(|e| format!("CUDA: {e}"))?;
    // SAFETY: loads the NVRTC library, as cudarc's NVRTC calls do first.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(
            "CUDA: NVRTC (libnvrtc, from the CUDA toolkit), which compiles the kernels, not found"
                .into(),
        );
    }
    if !cuda_path().is_some_and(|p| p.join("include/cuda_fp16.h").is_file()) {
        return Err("CUDA: the toolkit's headers (include/cuda_fp16.h) not found under CUDA_PATH, /usr/local/cuda, /opt/cuda or /usr".into());
    }
    Ok(Choice::Cuda {
        name,
        max_buffer: memory as u64 / 4,
    })
}

#[cfg(not(feature = "local-embed-cuda"))]
fn cuda() -> Result<Choice, String> {
    Err("CUDA needs arko_index built with the local-embed-cuda feature".into())
}

/// Where cubecl looks for the CUDA toolkit's headers, which NVRTC includes: `CUDA_PATH`,
/// else `/usr/local/cuda`, `/opt/cuda` or `/usr` (with `nvcc`) on Linux.
#[cfg(feature = "local-embed-cuda")]
fn cuda_path() -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Ok(path) = std::env::var("CUDA_PATH") {
        return Some(PathBuf::from(path));
    }
    if cfg!(target_os = "linux") {
        if Path::new("/usr/local/cuda").exists() {
            return Some("/usr/local/cuda".into());
        }
        if Path::new("/opt/cuda").exists() {
            return Some("/opt/cuda".into());
        }
        if Path::new("/usr/bin/nvcc").exists() {
            return Some("/usr".into());
        }
        return None;
    }
    cfg!(windows).then(|| "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/".into())
}

/// Which wgpu adapters a device choice takes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Adapters {
    Discrete,
    Gpus,
    Any,
}

/// The first wgpu adapter of the platform's graphics API that `adapters` allows:
/// discrete GPUs first, then integrated and virtual GPUs, then a CPU adapter.
#[cfg(feature = "local-embed-gpu")]
fn wgpu(adapters: Adapters) -> Result<Choice, String> {
    use burn_wgpu::{
        WgpuDevice,
        graphics::{AutoGraphicsApi, GraphicsApi},
    };
    use wgpu::DeviceType;
    let backend = AutoGraphicsApi::backend();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    // Adapters numbered within their type, as cubecl selects them.
    let mut found = Vec::new();
    let mut seen = [0usize; 4];
    for adapter in block_on(instance.enumerate_adapters(backend.into())) {
        let info = adapter.get_info();
        let (rank, device, allowed) = match info.device_type {
            DeviceType::DiscreteGpu => (0, WgpuDevice::DiscreteGpu(seen[0]), Adapters::Discrete),
            DeviceType::IntegratedGpu => (1, WgpuDevice::IntegratedGpu(seen[1]), Adapters::Gpus),
            DeviceType::VirtualGpu => (2, WgpuDevice::VirtualGpu(seen[2]), Adapters::Gpus),
            // cubecl takes the first CPU adapter.
            DeviceType::Cpu if seen[3] == 0 => (3, WgpuDevice::Cpu, Adapters::Any),
            _ => continue,
        };
        seen[rank] += 1;
        if allowed > adapters {
            continue;
        }
        let max_buffer = adapter.limits().max_storage_buffer_binding_size;
        found.push((rank, device, info, max_buffer));
    }
    found.sort_by_key(|f| f.0);
    match found.into_iter().next() {
        Some((_, device, info, max_buffer)) => Ok(Choice::Wgpu {
            device,
            name: format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type),
            max_buffer,
        }),
        None => Err(match adapters {
            Adapters::Discrete => "wgpu: no discrete GPU".into(),
            Adapters::Gpus => format!("wgpu: no GPU among the {backend:?} adapters"),
            Adapters::Any => format!("wgpu: no {backend:?} adapter"),
        }),
    }
}

#[cfg(not(feature = "local-embed-gpu"))]
fn wgpu(_: Adapters) -> Result<Choice, String> {
    Err("wgpu needs arko_index built with the local-embed-gpu feature".into())
}

/// Polls a future that completes without waiting, as wgpu's native ones do.
#[cfg(feature = "local-embed-gpu")]
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::task::{Context, Poll, Waker};
    let mut f = std::pin::pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

/// One loaded model on a GPU.
pub(crate) struct Gpu {
    run: Box<dyn Run>,
    name: String,
    hidden: usize,
    /// The most tokens one forward pass may hold on this device: one hidden state of
    /// them fills the largest buffer.
    max_pass_tokens: usize,
    /// Why the device failed, after a panic in its runtime, which leaves it unusable.
    failed: std::sync::Mutex<Option<String>>,
}

/// Forward passes of a loaded encoder, whatever its backend.
trait Run: Send + Sync {
    /// The pooled, unnormalized vectors of every batch's texts, in order. A batch is
    /// its texts and the length they are padded to.
    fn run(&self, batches: &[(Vec<&[u32]>, usize)]) -> Result<Vec<Vec<f32>>, Error>;
}

struct Runner<B: Backend> {
    encoder: Encoder<B>,
}

impl<B: Backend> Run for Runner<B>
where
    Encoder<B>: Send + Sync,
{
    fn run(&self, batches: &[(Vec<&[u32]>, usize)]) -> Result<Vec<Vec<f32>>, Error> {
        let h = self.encoder.hidden();
        let mut pending = Vec::with_capacity(batches.len());
        for (texts, len) in batches {
            pending.push(self.encoder.forward(texts, *len)?);
        }
        let mut out = Vec::new();
        for pooled in pending {
            let values: Vec<f32> = pooled
                .into_data()
                .to_vec()
                .map_err(|e| format!("reading the vectors back: {e:?}"))?;
            out.extend(values.chunks_exact(h).map(<[f32]>::to_vec));
        }
        Ok(out)
    }
}

/// The text of a panic's payload.
fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown error".into())
}

fn load_on<B: Backend>(
    device: B::Device,
    cfg: Config,
    pooling: Pooling,
    t: &Tensors,
    precision: Precision,
    max_elements: usize,
) -> Result<Box<dyn Run>, Error>
where
    Encoder<B>: Send + Sync,
{
    let (dtype, kind) = match precision {
        Precision::F32 => (FloatDType::F32, DType::F32),
        Precision::F16 => (FloatDType::F16, DType::F16),
        Precision::Bf16 => (FloatDType::BF16, DType::BF16),
        Precision::Int8 => {
            return Err("int8 runs on the CPU: a GPU runs f32, f16 or bf16".into());
        }
    };
    if !B::supports_dtype(&device, kind) {
        return Err(format!("this device has no {precision} arithmetic").into());
    }
    Ok(Box::new(Runner::<B> {
        encoder: Encoder::load(cfg, pooling, t, dtype, &device, max_elements)?,
    }))
}

/// cubecl saves its kernel tuning results under the nearest directory above the
/// working directory with a `Cargo.toml` (in its `target/`), else in the user's cache
/// directory. The embedder puts them in the user's cache directory (`cubecl/`) whatever
/// the working directory, unless the program configured cubecl before. Tests tune less.
fn configure_cubecl() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
        if CubeClRuntimeConfig::storage().lock().is_some() {
            return;
        }
        let mut config = CubeClRuntimeConfig::default().override_from_env();
        if let Some(dir) = user_cache_dir() {
            config.autotune.cache = CacheConfig::File(dir.join("cubecl"));
        }
        #[cfg(test)]
        {
            config.autotune.level = cubecl_runtime::config::autotune::AutotuneLevel::Minimal;
        }
        // Another thread may have read the configuration since the check.
        let _ = catch_unwind(|| CubeClRuntimeConfig::set(config));
    });
}

/// The user's cache directory: `$XDG_CACHE_HOME` or `~/.cache` on Linux and other Unix
/// systems, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows.
fn user_cache_dir() -> Option<std::path::PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from)
    };
    if cfg!(windows) {
        env("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches"))
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))
    }
}

impl Gpu {
    /// Loads the encoder onto `choice` with its matrices in `precision`.
    pub(crate) fn load(
        choice: Choice,
        cfg: Config,
        pooling: Pooling,
        t: &Tensors,
        precision: Precision,
    ) -> Result<Self, Error> {
        let hidden = cfg.hidden;
        let name = choice.name();
        // The most f32 values one tensor may hold.
        let max_elements = usize::try_from(choice.max_buffer() / 4).unwrap_or(usize::MAX);
        configure_cubecl();
        let loaded = catch_unwind(AssertUnwindSafe(|| -> Result<Box<dyn Run>, Error> {
            match choice {
                #[cfg(feature = "local-embed-cuda")]
                Choice::Cuda { .. } => load_on::<burn_cuda::Cuda<f32, i32>>(
                    burn_cuda::CudaDevice::default(),
                    cfg,
                    pooling,
                    t,
                    precision,
                    max_elements,
                ),
                #[cfg(feature = "local-embed-gpu")]
                Choice::Wgpu { device, .. } => load_on::<burn_wgpu::Wgpu<f32, i32, u32>>(
                    device,
                    cfg,
                    pooling,
                    t,
                    precision,
                    max_elements,
                ),
                #[cfg(test)]
                Choice::NdArray { .. } => load_on::<burn_ndarray::NdArray<f32>>(
                    Default::default(),
                    cfg,
                    pooling,
                    t,
                    precision,
                    max_elements,
                ),
            }
        }));
        let run = loaded.unwrap_or_else(|p| {
            Err(format!("loading the model on the GPU: {}", panic_text(p)).into())
        })?;
        Ok(Self {
            run,
            name,
            hidden,
            max_pass_tokens: (max_elements / hidden).max(PAD),
            failed: Default::default(),
        })
    }

    pub(crate) fn hidden(&self) -> usize {
        self.hidden
    }

    /// The device, for messages: `CUDA: NAME` or `wgpu: NAME (API, type)`.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Each sequence's unit vector, in order, from forward passes of at most
    /// `batch_tokens` padded tokens (and at most what the device's buffers hold).
    /// Returns the passes and their padded tokens too. After a panic in the GPU's
    /// runtime every call fails with its message.
    pub(crate) fn embed(
        &self,
        seqs: &[&[u32]],
        batch_tokens: usize,
    ) -> Result<(Vec<Vec<f32>>, Plan), Error> {
        let failed = || self.failed.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(why) = &*failed() {
            return Err(format!("the GPU failed before: {why}").into());
        }
        let lens: Vec<usize> = seqs.iter().map(|s| s.len()).collect();
        let plan = Plan::new(&lens, batch_tokens.min(self.max_pass_tokens));
        let batches: Vec<(Vec<&[u32]>, usize)> = plan
            .batches
            .iter()
            .map(|(texts, len)| (texts.iter().map(|&i| seqs[i]).collect(), *len))
            .collect();
        let pooled = match catch_unwind(AssertUnwindSafe(|| self.run.run(&batches))) {
            Ok(pooled) => pooled?,
            Err(p) => {
                let why = panic_text(p);
                *failed() = Some(why.clone());
                return Err(format!("the GPU failed: {why}").into());
            }
        };
        if pooled.len() != seqs.len() {
            return Err("the GPU returned the wrong number of vectors".into());
        }
        let mut out = vec![Vec::new(); seqs.len()];
        for (&i, mut v) in plan.order().zip(pooled) {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            out[i] = v;
        }
        Ok((out, plan))
    }
}

/// How a request's texts split into padded batches.
pub(crate) struct Plan {
    /// Each batch's texts (indices into the request) and padded length.
    pub(crate) batches: Vec<(Vec<usize>, usize)>,
}

impl Plan {
    /// Texts by token count (`lens`), longest first (ties in input order), in batches
    /// of at most `budget` padded tokens: a batch of rows padded to its first text's
    /// length, rounded up to a multiple of [`PAD`].
    pub(crate) fn new(lens: &[usize], budget: usize) -> Self {
        let mut order: Vec<usize> = (0..lens.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
        let mut batches = Vec::new();
        let mut k = 0;
        while k < order.len() {
            let len = lens[order[k]].div_ceil(PAD) * PAD;
            let rows = (budget / len).clamp(1, order.len() - k);
            batches.push((order[k..k + rows].to_vec(), len));
            k += rows;
        }
        Self { batches }
    }

    /// The texts in the order the batches hold them.
    fn order(&self) -> impl Iterator<Item = &usize> {
        self.batches.iter().flat_map(|(texts, _)| texts)
    }

    /// Tokens the forward passes computed, padding included.
    pub(crate) fn padded_tokens(&self) -> u64 {
        self.batches
            .iter()
            .map(|(texts, len)| (texts.len() * len) as u64)
            .sum()
    }
}

#[cfg(test)]
pub(crate) mod tests;
