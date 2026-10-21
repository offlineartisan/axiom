//! Injectable clock, so rate limiting and retry backoff can be tested without real sleeps

use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;

/// Source of time for rate limiting and retry backoff
#[async_trait]
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// The current instant on this clock
    fn now(&self) -> Instant;

    /// Wait for `duration` on this clock
    async fn sleep(&self, duration: Duration);
}

/// Clock backed by the operating system and the tokio timer
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

#[async_trait]
impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}

/// Manually advanced clock for tests.
///
/// `sleep` never blocks: it advances virtual time by the requested duration immediately and
/// records that duration so tests can assert on the backoff schedule.
#[derive(Debug)]
pub struct FakeClock {
    origin: Instant,
    elapsed: Mutex<Duration>,
    sleeps: Mutex<Vec<Duration>>,
}

impl FakeClock {
    /// Create a fake clock at virtual time zero
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            elapsed: Mutex::new(Duration::ZERO),
            sleeps: Mutex::new(Vec::new()),
        }
    }

    /// Advance virtual time without recording a sleep
    pub fn advance(&self, duration: Duration) {
        let mut elapsed = self.elapsed.lock().unwrap();
        *elapsed = elapsed.saturating_add(duration);
    }

    /// Every duration passed to `sleep`, in call order
    pub fn sleeps(&self) -> Vec<Duration> {
        self.sleeps.lock().unwrap().clone()
    }
}

impl Default for FakeClock {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Clock for FakeClock {
    fn now(&self) -> Instant {
        self.origin + *self.elapsed.lock().unwrap()
    }

    async fn sleep(&self, duration: Duration) {
        self.sleeps.lock().unwrap().push(duration);
        self.advance(duration);
    }
}
