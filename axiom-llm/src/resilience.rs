//! Resilient provider calls: rate limiting, retry with backoff, and metrics, applied per attempt.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio::time::timeout;

use axiom_ai_core::{AxiomError, Result};

use crate::clock::{Clock, SystemClock};
use crate::monitoring::LlmMetrics;
use crate::rate_limit::RateLimitConfig;
use crate::retry::RetryConfig;

/// Runs provider operations with the rate limit, retry policy and metrics applied to every attempt.
///
/// For each attempt, [`call`](Self::call):
/// 1. acquires the provider's rate limit (a rejection counts as a failed attempt),
/// 2. runs the operation, bounded by the attempt timeout,
/// 3. records the success or failure in [`LlmMetrics`],
/// 4. if the error is retryable and retries remain, waits the backoff on the [`Clock`] and tries again.
pub struct ResilientCall {
    retry: RetryConfig,
    rate_limits: RateLimitConfig,
    clock: Arc<dyn Clock>,
    metrics: Arc<RwLock<LlmMetrics>>,
}

impl ResilientCall {
    /// Use a different clock for rate limiting and backoff
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Set the retry policy
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry = config;
        self
    }

    /// Set the rate limits
    pub fn with_rate_limit_config(mut self, config: RateLimitConfig) -> Self {
        self.rate_limits = config;
        self
    }

    /// Shared metrics, updated once per attempt
    pub fn metrics(&self) -> &Arc<RwLock<LlmMetrics>> {
        &self.metrics
    }

    /// Run `op` on behalf of `provider`, retrying as the policy allows.
    ///
    /// `tokens` is the estimated token cost of one attempt, charged against the provider's token limit.
    pub async fn call<T, F, Fut>(&self, provider: &str, tokens: u32, mut op: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut retries_used = 0;
        loop {
            let started = self.clock.now();
            let outcome = match self
                .rate_limits
                .check_rate_limit_with_tokens(provider, started, tokens)
                .await
            {
                Ok(()) => self.attempt(&mut op).await,
                Err(rejected) => Err(rejected),
            };
            let latency = self.clock.now().saturating_duration_since(started);

            match outcome {
                Ok(value) => {
                    self.metrics.write().await.record_success(provider, latency);
                    return Ok(value);
                }
                Err(error) => {
                    self.metrics.write().await.record_failure(provider, latency);

                    if !error.is_retryable() || retries_used >= self.retry.max_retries {
                        return Err(error);
                    }

                    self.clock.sleep(self.retry.backoff_for(retries_used)).await;
                    retries_used += 1;
                }
            }
        }
    }

    /// Run one attempt, bounded by the attempt timeout when one is configured
    async fn attempt<T, F, Fut>(&self, op: &mut F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        match self.retry.attempt_timeout {
            Some(limit) => match timeout(limit, op()).await {
                Ok(result) => result,
                Err(_) => Err(AxiomError::Timeout(format!(
                    "Operation timed out after {:?}",
                    limit
                ))),
            },
            None => op().await,
        }
    }
}

impl Default for ResilientCall {
    fn default() -> Self {
        Self {
            retry: RetryConfig::default(),
            rate_limits: RateLimitConfig::new(),
            clock: Arc::new(SystemClock),
            metrics: Arc::new(RwLock::new(LlmMetrics::new())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FakeClock;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    fn unavailable() -> AxiomError {
        AxiomError::ProviderUnavailable("503 Service Unavailable".to_string())
    }

    fn network_error() -> AxiomError {
        // An invalid URL fails in the builder, so no network access happens.
        let err = reqwest::Client::new()
            .get("not a url")
            .build()
            .expect_err("invalid URL should fail to build");
        AxiomError::from(err)
    }

    /// Retry policy with a 100 ms first backoff so schedules are easy to read
    fn quick_retries(max_retries: u32) -> RetryConfig {
        RetryConfig::new(max_retries).with_initial_delay(Duration::from_millis(100))
    }

    fn call_with(clock: &Arc<FakeClock>, retry: RetryConfig) -> ResilientCall {
        ResilientCall::default()
            .with_clock(clock.clone())
            .with_retry_config(retry)
    }

    #[tokio::test]
    async fn transient_error_retries_up_to_max_attempts_with_recorded_backoff() {
        let clock = Arc::new(FakeClock::new());
        let call = call_with(&clock, quick_retries(3));
        let calls = AtomicU32::new(0);

        let result: Result<()> = call
            .call("p", 0, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(unavailable()) }
            })
            .await;

        assert!(matches!(result, Err(AxiomError::ProviderUnavailable(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 4, "one attempt plus three retries");
        assert_eq!(
            clock.sleeps(),
            vec![
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(400),
            ]
        );
    }

    #[tokio::test]
    async fn transient_failures_followed_by_success_return_the_value() {
        let clock = Arc::new(FakeClock::new());
        let call = call_with(&clock, quick_retries(3));
        let calls = AtomicU32::new(0);

        let result = call
            .call("p", 0, || {
                let attempt = calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if attempt < 2 {
                        Err(unavailable())
                    } else {
                        Ok("done")
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), "done");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(clock.sleeps().len(), 2);
    }

    #[tokio::test]
    async fn network_errors_are_retried() {
        let clock = Arc::new(FakeClock::new());
        let call = call_with(&clock, quick_retries(2));
        let calls = AtomicU32::new(0);

        let result: Result<()> = call
            .call("p", 0, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(network_error()) }
            })
            .await;

        assert!(matches!(result, Err(AxiomError::Network(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn fatal_provider_error_is_not_retried() {
        let clock = Arc::new(FakeClock::new());
        let call = call_with(&clock, quick_retries(3));
        let calls = AtomicU32::new(0);

        let result: Result<()> = call
            .call("p", 0, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(AxiomError::LlmProvider("model not supported".to_string())) }
            })
            .await;

        assert!(matches!(result, Err(AxiomError::LlmProvider(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(clock.sleeps().is_empty());
    }

    #[tokio::test]
    async fn each_attempt_consumes_rate_limit_capacity() {
        let clock = Arc::new(FakeClock::new());
        let rate_limits = RateLimitConfig::new();
        // Two requests per minute. Retry attempts share that budget with the first attempt.
        rate_limits.add_limiter("p".to_string(), Some(2), None).await;
        let call = call_with(&clock, quick_retries(3)).with_rate_limit_config(rate_limits);
        let calls = AtomicU32::new(0);

        let result: Result<()> = call
            .call("p", 0, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err(unavailable()) }
            })
            .await;

        // Attempts 1 and 2 reach the operation. Attempts 3 and 4 are rejected by the limiter.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(matches!(result, Err(AxiomError::RateLimit(_))));

        let metrics = call.metrics().read().await;
        assert_eq!(metrics.get_provider_metrics("p").unwrap().total_requests, 4);
    }

    #[tokio::test]
    async fn failures_are_counted_for_the_provider_that_failed_only() {
        let clock = Arc::new(FakeClock::new());
        let call = call_with(&clock, quick_retries(0));

        let _: Result<()> = call
            .call("a", 0, || async { Err(AxiomError::LlmProvider("bad".to_string())) })
            .await;
        let _: Result<()> = call.call("b", 0, || async { Ok(()) }).await;

        let metrics = call.metrics().read().await;
        let a = metrics.get_provider_metrics("a").expect("a recorded");
        assert_eq!((a.failed_requests, a.successful_requests), (1, 0));

        let b = metrics.get_provider_metrics("b").expect("b recorded");
        assert_eq!((b.failed_requests, b.successful_requests), (0, 1));

        assert_eq!(metrics.failed_requests, 1);
    }

    #[tokio::test]
    async fn unlimited_rate_limit_does_not_panic_after_time_advances() {
        let clock = Arc::new(FakeClock::new());
        let call = call_with(&clock, quick_retries(0));

        for _ in 0..3 {
            clock.advance(Duration::from_secs(24 * 3600));
            let result = call.call("p", u32::MAX, || async { Ok(7) }).await;
            assert_eq!(result.unwrap(), 7);
        }
    }

    #[tokio::test]
    async fn attempt_timeout_is_reported_as_a_retryable_timeout() {
        let clock = Arc::new(FakeClock::new());
        let retry = quick_retries(0).with_attempt_timeout(Duration::from_millis(10));
        let call = call_with(&clock, retry);

        // An operation that never finishes, so only the attempt timeout can end it.
        let result: Result<()> = call
            .call("p", 0, || std::future::pending::<Result<()>>())
            .await;

        assert!(matches!(result, Err(AxiomError::Timeout(_))));
    }
}
