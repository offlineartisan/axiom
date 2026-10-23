//! Retry policy: how many times to retry and how long to back off between attempts.
//!
//! The retry loop itself lives in [`crate::resilience::ResilientCall`], which checks
//! [`AxiomError::is_retryable`](axiom_ai_core::AxiomError::is_retryable) on every failure.

use std::time::Duration;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retries after the first attempt (total attempts = `max_retries + 1`)
    pub max_retries: u32,
    /// Delay before the first retry
    pub initial_delay: Duration,
    /// Upper bound on any single backoff delay
    pub max_delay: Duration,
    /// Multiplier applied to the delay after each retry
    pub backoff_multiplier: f64,
    /// Random spread around each delay, as a fraction of it (0.1 means ±10%). Zero disables jitter.
    pub jitter_factor: f64,
    /// Wall-clock limit for each attempt, enforced by tokio. `None` disables it.
    pub attempt_timeout: Option<Duration>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            attempt_timeout: Some(Duration::from_secs(30)),
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Set the initial delay
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the maximum delay
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the backoff multiplier
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set the attempt timeout
    pub fn with_attempt_timeout(mut self, timeout: Duration) -> Self {
        self.attempt_timeout = Some(timeout);
        self
    }

    /// Set the jitter factor (0.0 disables jitter)
    pub fn with_jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = factor;
        self
    }

    /// Delay to wait before retry number `retry_index` (0-based): exponential, capped at `max_delay`.
    ///
    /// This is the un-jittered schedule. See [`delay_for`](Self::delay_for) for the delay actually used.
    pub fn backoff_for(&self, retry_index: u32) -> Duration {
        let factor = self.backoff_multiplier.powf(f64::from(retry_index));
        let seconds = (self.initial_delay.as_secs_f64() * factor).min(self.max_delay.as_secs_f64());
        Duration::from_secs_f64(seconds.max(0.0))
    }

    /// Delay before retry `retry_index`, spread by `jitter_factor` so callers that fail together don't retry together.
    ///
    /// `unit` is a random sample in `[0, 1)`: 0 gives the shortest delay in the spread, and values
    /// near 1 the longest. The result is capped at `max_delay`.
    pub fn delay_for(&self, retry_index: u32, unit: f64) -> Duration {
        let base = self.backoff_for(retry_index).as_secs_f64();
        let spread = base * self.jitter_factor.max(0.0);
        let offset = (unit.clamp(0.0, 1.0) * 2.0 - 1.0) * spread;
        let seconds = (base + offset).clamp(0.0, self.max_delay.as_secs_f64());
        Duration::from_secs_f64(seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_exponentially() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(10));

        assert_eq!(config.backoff_for(0), Duration::from_millis(100));
        assert_eq!(config.backoff_for(1), Duration::from_millis(200));
        assert_eq!(config.backoff_for(2), Duration::from_millis(400));
    }

    #[test]
    fn backoff_is_capped_at_max_delay_even_for_large_retry_indexes() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_millis(100))
            .with_max_delay(Duration::from_secs(1));

        assert_eq!(config.backoff_for(4), Duration::from_secs(1));
        assert_eq!(config.backoff_for(10_000), Duration::from_secs(1));
    }

    #[test]
    fn jitter_spreads_the_delay_around_the_backoff() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_millis(100))
            .with_jitter_factor(0.5);

        // Base for retry 1 is 200 ms, spread ±50%
        assert_eq!(config.delay_for(1, 0.0), Duration::from_millis(100));
        assert_eq!(config.delay_for(1, 0.5), Duration::from_millis(200));
        let high = config.delay_for(1, 0.999_999);
        assert!(high > Duration::from_millis(299) && high <= Duration::from_millis(300), "{high:?}");
    }

    #[test]
    fn zero_jitter_uses_the_plain_backoff() {
        let config = RetryConfig::new(5).with_jitter_factor(0.0);

        for index in 0..4 {
            assert_eq!(config.delay_for(index, 0.0), config.backoff_for(index));
            assert_eq!(config.delay_for(index, 0.9), config.backoff_for(index));
        }
    }

    #[test]
    fn jittered_delay_never_exceeds_max_delay() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_secs(10))
            .with_max_delay(Duration::from_secs(10))
            .with_jitter_factor(0.5);

        assert!(config.delay_for(3, 0.999) <= Duration::from_secs(10));
    }
}
