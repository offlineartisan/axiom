//! Metrics for LLM operations, kept per provider with an aggregate across all of them.
//!
//! Metrics are recorded per attempt by [`crate::resilience::ResilientCall`]: every attempt
//! is either a success or a failure, and token usage is added by the gateway after a
//! successful generation.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

const ONE_MINUTE: Duration = Duration::from_secs(60);
const ONE_HOUR: Duration = Duration::from_secs(3600);

/// Aggregate metrics for LLM operations, with a breakdown per provider
#[derive(Debug, Clone)]
pub struct LlmMetrics {
    /// Total number of attempts, across all providers
    pub total_requests: u64,
    /// Total number of successful attempts
    pub successful_requests: u64,
    /// Total number of failed attempts
    pub failed_requests: u64,
    /// Total tokens used
    pub total_tokens: u64,
    /// Total cost in USD
    pub total_cost: f64,
    /// Average attempt latency in milliseconds
    pub avg_response_time_ms: f64,
    /// Metrics for each provider, keyed by provider name
    pub provider_metrics: HashMap<String, ProviderMetrics>,
    /// Attempts per minute over the last minute, across all providers
    pub request_rate: f64,
    /// Failed attempts over the last minute, across all providers
    pub error_rate: f64,
    /// Last updated timestamp
    pub last_updated: Instant,
}

/// Metrics for a specific provider
#[derive(Debug, Clone)]
pub struct ProviderMetrics {
    /// Provider name
    pub provider_name: String,
    /// Total attempts for this provider
    pub total_requests: u64,
    /// Successful attempts for this provider
    pub successful_requests: u64,
    /// Failed attempts for this provider
    pub failed_requests: u64,
    /// Total tokens used by this provider
    pub total_tokens: u64,
    /// Total cost for this provider
    pub total_cost: f64,
    /// Average attempt latency for this provider
    pub avg_response_time_ms: f64,
    /// Last attempt time
    pub last_request_time: Option<Instant>,
    /// Attempt timestamps from the last hour, oldest first (for rate calculation)
    pub request_history: VecDeque<Instant>,
    /// Failed attempt timestamps from the last hour, oldest first (for error rate calculation)
    pub error_history: VecDeque<Instant>,
}

impl LlmMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_tokens: 0,
            total_cost: 0.0,
            avg_response_time_ms: 0.0,
            provider_metrics: HashMap::new(),
            request_rate: 0.0,
            error_rate: 0.0,
            last_updated: Instant::now(),
        }
    }

    /// Record a successful attempt against `provider_name`
    pub fn record_success(&mut self, provider_name: &str, latency: Duration) {
        self.record_attempt(provider_name, latency, true);
    }

    /// Record a failed attempt against `provider_name`
    pub fn record_failure(&mut self, provider_name: &str, latency: Duration) {
        self.record_attempt(provider_name, latency, false);
    }

    /// Add token usage and cost for `provider_name`. Does not count as an attempt.
    pub fn record_usage(&mut self, provider_name: &str, tokens: u64, cost: f64) {
        self.total_tokens = self.total_tokens.saturating_add(tokens);
        self.total_cost += cost;

        let provider = self.provider_entry(provider_name);
        provider.total_tokens = provider.total_tokens.saturating_add(tokens);
        provider.total_cost += cost;
    }

    fn record_attempt(&mut self, provider_name: &str, latency: Duration, success: bool) {
        let now = Instant::now();
        let latency_ms = latency.as_secs_f64() * 1000.0;

        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }
        self.avg_response_time_ms = running_average(self.avg_response_time_ms, self.total_requests, latency_ms);

        self.provider_entry(provider_name).record_attempt(now, latency_ms, success);

        self.update_rates(now);
        self.last_updated = now;
    }

    fn provider_entry(&mut self, provider_name: &str) -> &mut ProviderMetrics {
        self.provider_metrics
            .entry(provider_name.to_string())
            .or_insert_with(|| ProviderMetrics::new(provider_name.to_string()))
    }

    /// Recompute the aggregate per-minute rates from the provider histories
    fn update_rates(&mut self, now: Instant) {
        self.request_rate = self
            .provider_metrics
            .values()
            .map(|pm| count_within(&pm.request_history, now, ONE_MINUTE))
            .sum::<usize>() as f64;
        self.error_rate = self
            .provider_metrics
            .values()
            .map(|pm| count_within(&pm.error_history, now, ONE_MINUTE))
            .sum::<usize>() as f64;
    }

    /// Get metrics for a specific provider
    pub fn get_provider_metrics(&self, provider_name: &str) -> Option<&ProviderMetrics> {
        self.provider_metrics.get(provider_name)
    }

    /// Get all provider names
    pub fn get_provider_names(&self) -> Vec<&String> {
        self.provider_metrics.keys().collect()
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for LlmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderMetrics {
    /// Create new provider metrics
    pub fn new(provider_name: String) -> Self {
        Self {
            provider_name,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_tokens: 0,
            total_cost: 0.0,
            avg_response_time_ms: 0.0,
            last_request_time: None,
            request_history: VecDeque::new(),
            error_history: VecDeque::new(),
        }
    }

    fn record_attempt(&mut self, now: Instant, latency_ms: f64, success: bool) {
        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
            self.error_history.push_back(now);
        }
        self.avg_response_time_ms = running_average(self.avg_response_time_ms, self.total_requests, latency_ms);

        self.last_request_time = Some(now);
        self.request_history.push_back(now);

        // Timestamps are pushed in time order, so expired ones are at the front
        drop_older_than(&mut self.request_history, now, ONE_HOUR);
        drop_older_than(&mut self.error_history, now, ONE_HOUR);
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.failed_requests as f64 / self.total_requests as f64
        }
    }

    /// Attempts in the last minute
    pub fn requests_per_minute(&self) -> f64 {
        count_within(&self.request_history, Instant::now(), ONE_MINUTE) as f64
    }

    /// Failed attempts in the last minute
    pub fn errors_per_minute(&self) -> f64 {
        count_within(&self.error_history, Instant::now(), ONE_MINUTE) as f64
    }
}

/// Remove timestamps at least `window` older than `now`. `times` must be oldest first.
fn drop_older_than(times: &mut VecDeque<Instant>, now: Instant, window: Duration) {
    while times.front().is_some_and(|&t| now.saturating_duration_since(t) >= window) {
        times.pop_front();
    }
}

/// Number of timestamps no older than `window` at `now`. `times` must be oldest first.
fn count_within(times: &VecDeque<Instant>, now: Instant, window: Duration) -> usize {
    let expired = times.partition_point(|&t| now.saturating_duration_since(t) >= window);
    times.len() - expired
}

/// Incremental mean: folds `sample` into `average`, where `count` is the number of samples including this one
fn running_average(average: f64, count: u64, sample: f64) -> f64 {
    average + (sample - average) / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_are_counted_for_the_provider_that_failed_only() {
        let mut metrics = LlmMetrics::new();
        metrics.record_failure("a", Duration::from_millis(10));
        metrics.record_success("b", Duration::from_millis(10));

        let a = metrics.get_provider_metrics("a").expect("provider a recorded");
        assert_eq!(a.failed_requests, 1);
        assert_eq!(a.successful_requests, 0);

        let b = metrics.get_provider_metrics("b").expect("provider b recorded");
        assert_eq!(b.failed_requests, 0);
        assert_eq!(b.successful_requests, 1);

        assert_eq!(metrics.total_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
    }

    #[test]
    fn usage_is_added_without_counting_an_attempt() {
        let mut metrics = LlmMetrics::new();
        metrics.record_success("a", Duration::ZERO);
        metrics.record_usage("a", 120, 0.5);

        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.total_tokens, 120);
        let a = metrics.get_provider_metrics("a").unwrap();
        assert_eq!(a.total_tokens, 120);
        assert!((a.total_cost - 0.5).abs() < f64::EPSILON);
    }
}
