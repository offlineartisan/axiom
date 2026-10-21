//! Rate limiting implementation
//!
//! Limits are per provider. A provider with no configured limit is not limited at all:
//! there is no implicit default. Time is passed in explicitly (usually from a
//! [`Clock`](crate::clock::Clock)) so tests can drive it deterministically.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use axiom_ai_core::{Result, AxiomError};

/// Token bucket that refills continuously at `refill_per_sec` up to `capacity`.
///
/// The bucket starts full. It only refills while time is observed through its methods.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_per_sec: f64,
    last_refill: Option<Instant>,
}

impl TokenBucket {
    /// Create a full bucket. A `refill_per_sec` of zero means it never refills.
    pub fn new(capacity: u32, refill_per_sec: f64) -> Self {
        Self {
            capacity: f64::from(capacity),
            tokens: f64::from(capacity),
            refill_per_sec,
            last_refill: None,
        }
    }

    /// Add the tokens earned since the last observation, capped at capacity
    fn refill(&mut self, now: Instant) {
        if let Some(last) = self.last_refill {
            let elapsed = now.saturating_duration_since(last).as_secs_f64();
            self.tokens = (self.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        }
        self.last_refill = Some(now);
    }

    /// Try to take `tokens` from the bucket. Returns `false` and takes nothing if not enough are available.
    pub fn try_consume(&mut self, now: Instant, tokens: u32) -> bool {
        self.refill(now);
        let needed = f64::from(tokens);
        if self.tokens >= needed {
            self.tokens -= needed;
            true
        } else {
            false
        }
    }

    /// Whole tokens currently available
    pub fn available_tokens(&mut self, now: Instant) -> u32 {
        self.refill(now);
        // Non-negative and at most `capacity` (a u32), so the cast cannot truncate.
        self.tokens.floor() as u32
    }

    /// Time until `tokens` could be taken. Zero if they are available now, `Duration::MAX` if the bucket never refills.
    pub fn time_until(&mut self, now: Instant, tokens: u32) -> Duration {
        self.refill(now);
        let missing = f64::from(tokens) - self.tokens;
        if missing <= 0.0 {
            Duration::ZERO
        } else if self.refill_per_sec > 0.0 {
            Duration::from_secs_f64(missing / self.refill_per_sec)
        } else {
            Duration::MAX
        }
    }
}

/// Rate limiter for a single provider: optional request and token buckets
#[derive(Debug)]
pub struct RateLimiter {
    /// Bucket for requests, absent when requests are unlimited
    requests: Option<TokenBucket>,
    /// Bucket for tokens, absent when tokens are unlimited
    tokens: Option<TokenBucket>,
    /// Maximum requests per minute
    requests_per_minute: Option<u32>,
    /// Maximum tokens per minute
    tokens_per_minute: Option<u32>,
}

impl RateLimiter {
    /// Create a rate limiter. `None` for a dimension means it is unlimited.
    pub fn new(requests_per_minute: Option<u32>, tokens_per_minute: Option<u32>) -> Self {
        Self {
            requests: requests_per_minute.map(|rpm| TokenBucket::new(rpm, per_second(rpm))),
            tokens: tokens_per_minute.map(|tpm| TokenBucket::new(tpm, per_second(tpm))),
            requests_per_minute,
            tokens_per_minute,
        }
    }

    /// Admit one request that will use `token_count` tokens.
    ///
    /// Both buckets are checked before either is charged, so a rejected request costs nothing.
    /// The token cost is capped at the tokens-per-minute limit, so an oversized request is
    /// admitted once the bucket is full instead of waiting forever. On rejection, returns how
    /// long to wait.
    pub fn try_acquire(&mut self, now: Instant, token_count: u32) -> std::result::Result<(), Duration> {
        let token_cost = match self.tokens_per_minute {
            Some(tpm) => token_count.min(tpm),
            None => token_count,
        };

        let wait_requests = self
            .requests
            .as_mut()
            .map_or(Duration::ZERO, |bucket| bucket.time_until(now, 1));
        let wait_tokens = self
            .tokens
            .as_mut()
            .map_or(Duration::ZERO, |bucket| bucket.time_until(now, token_cost));

        let wait = wait_requests.max(wait_tokens);
        if !wait.is_zero() {
            return Err(wait);
        }

        // Both buckets were checked above, so these charges succeed.
        if let Some(bucket) = self.requests.as_mut() {
            bucket.try_consume(now, 1);
        }
        if let Some(bucket) = self.tokens.as_mut() {
            bucket.try_consume(now, token_cost);
        }
        Ok(())
    }

    /// Snapshot of the limiter at `now`
    pub fn status(&mut self, now: Instant) -> RateLimitStatus {
        RateLimitStatus {
            requests_per_minute: self.requests_per_minute,
            tokens_per_minute: self.tokens_per_minute,
            available_requests: self.requests.as_mut().map(|bucket| bucket.available_tokens(now)),
            available_tokens: self.tokens.as_mut().map(|bucket| bucket.available_tokens(now)),
            time_until_next_request: self
                .requests
                .as_mut()
                .map_or(Duration::ZERO, |bucket| bucket.time_until(now, 1)),
        }
    }
}

/// Convert a per-minute limit to a per-second refill rate
fn per_second(per_minute: u32) -> f64 {
    f64::from(per_minute) / 60.0
}

/// Rate limits for every provider
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Rate limiters by provider name
    limiters: Arc<RwLock<HashMap<String, RateLimiter>>>,
    /// Limits applied to providers without an explicit limiter. `None` leaves them unlimited.
    default_requests_per_minute: Option<u32>,
    /// Token limit applied to providers without an explicit limiter. `None` leaves them unlimited.
    default_tokens_per_minute: Option<u32>,
}

impl RateLimitConfig {
    /// Create a configuration with no limits: every provider is unlimited until one is configured
    pub fn new() -> Self {
        Self {
            limiters: Arc::new(RwLock::new(HashMap::new())),
            default_requests_per_minute: None,
            default_tokens_per_minute: None,
        }
    }

    /// Set the limits applied to providers without an explicit limiter
    pub fn with_default_limits(
        mut self,
        requests_per_minute: Option<u32>,
        tokens_per_minute: Option<u32>,
    ) -> Self {
        self.default_requests_per_minute = requests_per_minute;
        self.default_tokens_per_minute = tokens_per_minute;
        self
    }

    /// Add a rate limiter for a specific provider
    pub async fn add_limiter(
        &self,
        provider_name: String,
        requests_per_minute: Option<u32>,
        tokens_per_minute: Option<u32>,
    ) {
        let mut limiters = self.limiters.write().await;
        limiters.insert(
            provider_name,
            RateLimiter::new(requests_per_minute, tokens_per_minute),
        );
    }

    /// Check that one request to `provider_name` is allowed at `now`
    pub async fn check_rate_limit(&self, provider_name: &str, now: Instant) -> Result<()> {
        self.check_rate_limit_with_tokens(provider_name, now, 0).await
    }

    /// Check that one request using `token_count` tokens is allowed at `now`
    pub async fn check_rate_limit_with_tokens(
        &self,
        provider_name: &str,
        now: Instant,
        token_count: u32,
    ) -> Result<()> {
        let mut limiters = self.limiters.write().await;

        if !limiters.contains_key(provider_name) {
            // No explicit limiter and no default limits: nothing to enforce.
            if self.default_requests_per_minute.is_none() && self.default_tokens_per_minute.is_none() {
                return Ok(());
            }
            limiters.insert(
                provider_name.to_string(),
                RateLimiter::new(self.default_requests_per_minute, self.default_tokens_per_minute),
            );
        }

        match limiters.get_mut(provider_name) {
            Some(limiter) => limiter.try_acquire(now, token_count).map_err(|wait| {
                AxiomError::RateLimit(format!(
                    "Rate limit exceeded for provider '{}' with {} tokens. Try again in {:?}",
                    provider_name, token_count, wait
                ))
            }),
            None => Ok(()),
        }
    }

    /// Get rate limit status for a provider at `now`. `None` if the provider has no limiter.
    pub async fn get_status(&self, provider_name: &str, now: Instant) -> Option<RateLimitStatus> {
        let mut limiters = self.limiters.write().await;
        limiters.get_mut(provider_name).map(|limiter| limiter.status(now))
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limit status for a provider
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    /// Maximum requests per minute, `None` if unlimited
    pub requests_per_minute: Option<u32>,
    /// Maximum tokens per minute, `None` if unlimited
    pub tokens_per_minute: Option<u32>,
    /// Requests available right now, `None` if unlimited
    pub available_requests: Option<u32>,
    /// Tokens available right now, `None` if unlimited
    pub available_tokens: Option<u32>,
    /// Time until the next request is allowed
    pub time_until_next_request: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{Clock, FakeClock};

    #[test]
    fn token_bucket_refills_over_time() {
        let clock = FakeClock::new();
        let mut bucket = TokenBucket::new(10, 1.0); // 10 tokens, 1 per second

        assert!(bucket.try_consume(clock.now(), 10));
        assert!(!bucket.try_consume(clock.now(), 1));

        clock.advance(Duration::from_secs(1));
        assert!(bucket.try_consume(clock.now(), 1));
        assert!(!bucket.try_consume(clock.now(), 1));
    }

    #[test]
    fn token_bucket_keeps_fractional_refill_and_caps_at_capacity() {
        let clock = FakeClock::new();
        let mut bucket = TokenBucket::new(10, 1.0);
        assert!(bucket.try_consume(clock.now(), 10));

        // Half a second at a time: the fractions must add up, not be dropped.
        clock.advance(Duration::from_millis(500));
        assert_eq!(bucket.available_tokens(clock.now()), 0);
        clock.advance(Duration::from_millis(500));
        assert_eq!(bucket.available_tokens(clock.now()), 1);

        // A long idle period refills to capacity and no further.
        clock.advance(Duration::from_secs(3600));
        assert_eq!(bucket.available_tokens(clock.now()), 10);
    }

    #[test]
    fn time_until_reports_the_wait_for_missing_tokens() {
        let clock = FakeClock::new();
        let mut bucket = TokenBucket::new(1, 2.0); // 2 tokens per second
        assert!(bucket.try_consume(clock.now(), 1));
        assert_eq!(bucket.time_until(clock.now(), 1), Duration::from_millis(500));
    }

    #[test]
    fn unlimited_limiter_never_rejects_after_time_advances() {
        let clock = FakeClock::new();
        let mut limiter = RateLimiter::new(None, None);

        for _ in 0..3 {
            clock.advance(Duration::from_secs(3600));
            assert!(limiter.try_acquire(clock.now(), u32::MAX).is_ok());
        }
    }

    #[test]
    fn limiter_charges_both_buckets_only_when_admitted() {
        let clock = FakeClock::new();
        let mut limiter = RateLimiter::new(Some(60), Some(100));

        assert!(limiter.try_acquire(clock.now(), 100).is_ok());
        // Token bucket is empty, so this is rejected and must not charge the request bucket.
        assert!(limiter.try_acquire(clock.now(), 1).is_err());

        let status = limiter.status(clock.now());
        assert_eq!(status.available_requests, Some(59));
        assert_eq!(status.available_tokens, Some(0));
    }

    #[test]
    fn oversized_token_cost_is_capped_at_the_token_limit() {
        let clock = FakeClock::new();
        let mut limiter = RateLimiter::new(None, Some(100));

        assert!(limiter.try_acquire(clock.now(), 10_000).is_ok());
        assert!(limiter.try_acquire(clock.now(), 10_000).is_err());
    }

    #[tokio::test]
    async fn only_configured_providers_are_limited() {
        let clock = FakeClock::new();
        let config = RateLimitConfig::new();
        config.add_limiter("limited".to_string(), Some(1), None).await;

        // No limiter and no defaults: unlimited.
        assert!(config.check_rate_limit("unconfigured", clock.now()).await.is_ok());
        assert!(config.check_rate_limit("unconfigured", clock.now()).await.is_ok());

        assert!(config.check_rate_limit("limited", clock.now()).await.is_ok());
        assert!(config.check_rate_limit("limited", clock.now()).await.is_err());
    }

    #[tokio::test]
    async fn default_limits_apply_to_providers_without_a_limiter() {
        let clock = FakeClock::new();
        let config = RateLimitConfig::new().with_default_limits(Some(1), None);

        assert!(config.check_rate_limit("anything", clock.now()).await.is_ok());
        assert!(config.check_rate_limit("anything", clock.now()).await.is_err());
    }

    #[tokio::test]
    async fn unlimited_config_does_not_panic_after_time_advances() {
        let clock = FakeClock::new();
        let config = RateLimitConfig::new();

        clock.advance(Duration::from_secs(7 * 24 * 3600));
        assert!(config
            .check_rate_limit_with_tokens("provider", clock.now(), u32::MAX)
            .await
            .is_ok());
    }
}
