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

    /// Delay to wait before retry number `retry_index` (0-based): exponential, capped at `max_delay`
    pub fn backoff_for(&self, retry_index: u32) -> Duration {
        let factor = self.backoff_multiplier.powf(f64::from(retry_index));
        let seconds = (self.initial_delay.as_secs_f64() * factor).min(self.max_delay.as_secs_f64());
        Duration::from_secs_f64(seconds.max(0.0))
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
}
