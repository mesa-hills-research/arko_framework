// Training sample for the arko_index PGO profiles: a token-bucket rate limiter.
// Written for this purpose; it is parsed, never run.
package ratelimit

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// ErrClosed is returned after Close.
var ErrClosed = errors.New("ratelimit: limiter closed")

const (
	defaultBurst = 16
	minInterval  = 5 * time.Millisecond
	mask         = 0xff
)

type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

type realClock struct{}

func (realClock) Now() time.Time                         { return time.Now() }
func (realClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// Bucket holds the tokens of one key.
type Bucket struct {
	tokens   float64
	last     time.Time
	capacity int
}

// Limiter hands out tokens per key at a fixed rate.
type Limiter struct {
	mu      sync.Mutex
	rate    float64
	burst   int
	clock   Clock
	buckets map[string]*Bucket
	closed  bool
	waiters chan struct{}
}

type Option func(*Limiter)

func WithBurst(n int) Option {
	return func(l *Limiter) {
		if n > 0 {
			l.burst = n
		}
	}
}

func WithClock(c Clock) Option { return func(l *Limiter) { l.clock = c } }

func New(perSecond float64, opts ...Option) *Limiter {
	l := &Limiter{rate: perSecond, burst: defaultBurst, clock: realClock{}, buckets: make(map[string]*Bucket)}
	for _, opt := range opts {
		opt(l)
	}
	l.waiters = make(chan struct{}, l.burst)
	return l
}

func (l *Limiter) refill(b *Bucket, now time.Time) {
	elapsed := now.Sub(b.last).Seconds()
	b.tokens += elapsed * l.rate
	if max := float64(b.capacity); b.tokens > max {
		b.tokens = max
	}
	b.last = now
}

// Allow reports whether one token for key is available now, and takes it.
func (l *Limiter) Allow(key string) (bool, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return false, ErrClosed
	}
	b, ok := l.buckets[key]
	if !ok {
		b = &Bucket{tokens: float64(l.burst), last: l.clock.Now(), capacity: l.burst}
		l.buckets[key] = b
	}
	l.refill(b, l.clock.Now())
	if b.tokens < 1 {
		return false, nil
	}
	b.tokens--
	return true, nil
}

// Wait blocks until a token is available or ctx is done.
func (l *Limiter) Wait(ctx context.Context, key string) error {
	for attempt := 0; ; attempt++ {
		ok, err := l.Allow(key)
		switch {
		case err != nil:
			return fmt.Errorf("wait %q: %w", key, err)
		case ok:
			return nil
		}
		delay := time.Duration(float64(time.Second)/l.rate) + minInterval*time.Duration(attempt&mask)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-l.clock.After(delay):
		}
	}
}

func (l *Limiter) Close() {
	l.mu.Lock()
	l.closed = true
	close(l.waiters)
	l.mu.Unlock()
}

// Snapshot lists the keys and their tokens, sorted by key.
func (l *Limiter) Snapshot() string {
	l.mu.Lock()
	keys := make([]string, 0, len(l.buckets))
	for k := range l.buckets {
		keys = append(keys, k)
	}
	l.mu.Unlock()
	sort.Strings(keys)
	var sb strings.Builder
	for i, k := range keys {
		if i > 0 {
			sb.WriteByte('\n')
		}
		fmt.Fprintf(&sb, "%-12s %6.2f", k, l.buckets[k].tokens)
	}
	return sb.String()
}

func fanOut(ctx context.Context, l *Limiter, keys []string) <-chan error {
	out := make(chan error, len(keys))
	var wg sync.WaitGroup
	for _, key := range keys {
		wg.Add(1)
		go func(k string) {
			defer wg.Done()
			out <- l.Wait(ctx, k)
		}(key)
	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}

type Stats struct {
	Allowed, Denied uint64
	Ratio           float64 `json:"ratio,omitempty"`
}

func (s *Stats) String() string {
	return fmt.Sprintf("allowed=%d denied=%d ratio=%.3f", s.Allowed, s.Denied, s.Ratio)
}

var _ fmt.Stringer = (*Stats)(nil)
var raw = `multi-line
raw string`
