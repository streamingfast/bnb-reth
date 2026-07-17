//! Chain-specific tracing configuration and the thread-local active-tracer handle.
//!
//! Some chains route value differently from mainnet Ethereum in ways the generic
//! [`crate::executor::FirehoseWrappedExecutor`] / [`crate::inspector::FirehoseInspector`]
//! cannot infer from the transaction stream alone:
//!
//! * **BSC (Parlia)** credits every transaction's gas fee (and blob fee) to the consensus
//!   `SYSTEM_ADDRESS` (`0xffff…fffe`) instead of the block beneficiary, defers the consensus
//!   "system transactions" found in the block body to the end-of-block finalize step, and sweeps
//!   the accumulated fees from `SYSTEM_ADDRESS` to the validator with direct (non-EVM) state
//!   writes.
//!
//! A chain integrator registers a [`ChainTracingConfig`] once at process startup (next to
//! [`crate::init_tracer`]); the generic executor wrapper and inspector consult it at the
//! relevant decision points. When no config is registered, behavior is mainnet-Ethereum.
//!
//! The second half of this module is the **active-tracer handle**: the block tracer guard
//! ([`crate::FirehoseBlockTracer`]) holds the global tracer mutex for the whole block, so
//! chain executor code that runs *inside* block execution (e.g. BSC's `finish()`-time system
//! transactions) cannot re-lock the global. Instead, the guard registers a thread-local
//! pointer to the tracer for its lifetime, and chain code emits through
//! [`with_active_tracer`].

use alloy_primitives::Address;
use std::{cell::Cell, ptr::NonNull, sync::OnceLock};

/// Chain-specific tracing behavior overrides. See the module docs for background.
#[derive(Debug, Clone, Copy)]
pub struct ChainTracingConfig {
    /// Address credited with transaction fees instead of the block beneficiary
    /// (`None` = beneficiary, the mainnet Ethereum behavior).
    ///
    /// BSC: `SYSTEM_ADDRESS` (`0xffff…fffe`); the fee is swept to the validator later by the
    /// consensus engine.
    pub fee_recipient: Option<Address>,

    /// Emit a `REASON_REWARD_BLOB_FEE` balance change crediting the fee recipient with the
    /// EIP-4844 blob fee (`blob_gas_used × blob_gas_price`). Mainnet Ethereum burns the blob
    /// fee (no balance change); BSC credits it to `SYSTEM_ADDRESS`.
    pub reward_blob_fee: bool,

    /// Returns `true` when a transaction in the block body is a consensus "system transaction"
    /// that the chain's block executor defers: it is skipped during body iteration and executed
    /// by the consensus engine during `finish()`. For such transactions the generic wrapper
    /// emits **nothing** — the chain executor is responsible for emitting the
    /// `on_tx_start`/`on_tx_end` pair (via [`with_active_tracer`]) at actual execution time,
    /// which matches the geth reference where system txs are traced when the consensus engine
    /// applies them.
    ///
    /// Arguments: `to`, `max_fee_per_gas`, `signer`, `block_beneficiary`.
    pub is_deferred_system_tx: fn(Option<Address>, u128, Address, Address) -> bool,

    /// Wrap the inner executor's `finish()` in an `on_system_call_start`/`end` window.
    ///
    /// `true` (mainnet Ethereum): post-execution work in `finish()` consists of EVM system
    /// calls (EIP-7002/7251 request processing) that belong in `block.system_calls`.
    ///
    /// `false` (BSC): `finish()` runs Parlia finalization — deferred system *transactions* and
    /// direct reward sweeps — which the chain executor emits itself as transaction traces and
    /// block-level balance changes; a system-call window would swallow them.
    pub trace_finish_in_system_call: bool,
}

static CHAIN_TRACING: OnceLock<ChainTracingConfig> = OnceLock::new();

/// Registers the process-wide chain tracing config. Call at most once, at startup, before any
/// block is traced. Panics on double-registration.
pub fn set_chain_tracing_config(config: ChainTracingConfig) {
    CHAIN_TRACING.set(config).expect("set_chain_tracing_config called more than once");
}

/// Returns the registered chain tracing config, if any.
pub fn chain_tracing_config() -> Option<&'static ChainTracingConfig> {
    CHAIN_TRACING.get()
}

thread_local! {
    /// Pointer to the tracer owned by the currently active block guard on this thread, plus a
    /// borrow flag preventing re-entrant use. See [`with_active_tracer`].
    static ACTIVE_TRACER: Cell<Option<NonNull<firehose_tracer::Tracer>>> = const { Cell::new(None) };
    static ACTIVE_TRACER_BORROWED: Cell<bool> = const { Cell::new(false) };
}

/// Registers `tracer` as this thread's active tracer. Called by the block tracer guard on
/// construction; must be paired with [`clear_active_tracer`] (the guard's `Drop` does this).
pub(crate) fn set_active_tracer(tracer: &mut firehose_tracer::Tracer) {
    ACTIVE_TRACER.with(|slot| slot.set(Some(NonNull::from(tracer))));
}

/// Clears this thread's active tracer registration.
pub(crate) fn clear_active_tracer() {
    ACTIVE_TRACER.with(|slot| slot.set(None));
}

/// Runs `f` with mutable access to the tracer of the block currently being traced on this
/// thread, or returns `None` when no block guard is active (tracing disabled, or the caller is
/// on a different thread than block execution).
///
/// This is the emission path for chain executor code that runs *inside* block execution and
/// therefore cannot lock the global tracer (the block guard already holds it) — e.g. BSC's
/// end-of-block system transactions and reward sweeps.
///
/// # Safety model
///
/// The pointer aliases the `&mut Tracer` held by the block guard (and borrowed by the EVM
/// inspector during frame callbacks). Soundness rests on temporal exclusivity on a single
/// thread: chain executor code calls this between EVM operations, never from inside an
/// inspector callback. The `ACTIVE_TRACER_BORROWED` flag turns any accidental re-entrant use
/// into a clean `None` instead of aliasing UB.
pub fn with_active_tracer<R>(f: impl FnOnce(&mut firehose_tracer::Tracer) -> R) -> Option<R> {
    let ptr = ACTIVE_TRACER.with(|slot| slot.get())?;
    let already_borrowed = ACTIVE_TRACER_BORROWED.with(|b| b.replace(true));
    if already_borrowed {
        return None;
    }
    // SAFETY: the pointer was registered from a live `&mut Tracer` owned by the block guard on
    // this same thread; the guard's Drop clears the registration before the borrow ends. The
    // borrow flag above rejects re-entrant calls, and inspector callbacks never call this
    // function, so no other `&mut` to the tracer is active during `f`.
    let result = f(unsafe { &mut *ptr.as_ptr() });
    ACTIVE_TRACER_BORROWED.with(|b| b.set(false));
    Some(result)
}
