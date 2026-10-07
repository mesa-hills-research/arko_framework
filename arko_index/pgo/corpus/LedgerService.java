// Training sample for the arko_index PGO profiles: a small ledger service.
// Written for this purpose; it is parsed, never run.
package dev.arko.sample.ledger;

import java.io.IOException;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Keeps balances per account and an append-only list of entries.
 *
 * @param <K> the account key type
 */
public final class LedgerService<K extends Comparable<K>> implements AutoCloseable {
    public static final int MAX_ENTRIES = 1 << 16;
    private static final BigDecimal ZERO = BigDecimal.ZERO;

    public enum Kind { CREDIT, DEBIT, ADJUSTMENT }

    public record Entry<K>(K account, Kind kind, BigDecimal amount, Instant at, String memo) {
        public Entry {
            Objects.requireNonNull(account, "account");
            if (amount.signum() < 0) {
                throw new IllegalArgumentException("amount must not be negative: " + amount);
            }
        }
    }

    public static class InsufficientFundsException extends Exception {
        private final BigDecimal shortfall;

        public InsufficientFundsException(String message, BigDecimal shortfall) {
            super(message);
            this.shortfall = shortfall;
        }

        public BigDecimal getShortfall() {
            return shortfall;
        }
    }

    @FunctionalInterface
    public interface Listener<K> {
        void onEntry(Entry<K> entry);
    }

    private final Map<K, BigDecimal> balances = new ConcurrentHashMap<>();
    private final List<Entry<K>> entries = Collections.synchronizedList(new ArrayList<>());
    private final List<Listener<K>> listeners = new ArrayList<>();
    private volatile boolean closed;

    public LedgerService() {
        this(List.of());
    }

    public LedgerService(Collection<Listener<K>> initial) {
        listeners.addAll(initial);
    }

    public synchronized void record(K account, Kind kind, BigDecimal amount, String memo)
            throws InsufficientFundsException {
        if (closed) {
            throw new IllegalStateException("ledger is closed");
        }
        BigDecimal current = balances.getOrDefault(account, ZERO);
        BigDecimal next = switch (kind) {
            case CREDIT -> current.add(amount);
            case DEBIT -> {
                if (current.compareTo(amount) < 0) {
                    throw new InsufficientFundsException(
                            String.format("%s has %s, needs %s", account, current, amount),
                            amount.subtract(current));
                }
                yield current.subtract(amount);
            }
            case ADJUSTMENT -> amount;
        };
        balances.put(account, next);
        Entry<K> entry = new Entry<>(account, kind, amount, Instant.now(), memo);
        if (entries.size() >= MAX_ENTRIES) {
            entries.remove(0);
        }
        entries.add(entry);
        for (Listener<K> listener : listeners) {
            listener.onEntry(entry);
        }
    }

    public BigDecimal balance(K account) {
        return balances.getOrDefault(account, ZERO);
    }

    public List<Entry<K>> find(Predicate<? super Entry<K>> filter, int limit) {
        synchronized (entries) {
            return entries.stream()
                    .filter(filter)
                    .sorted(Comparator.comparing(Entry::at, Comparator.reverseOrder()))
                    .limit(limit)
                    .collect(Collectors.toList());
        }
    }

    public Map<Kind, Long> countByKind() {
        Map<Kind, Long> counts = new EnumMap<>(Kind.class);
        for (int i = 0; i < entries.size(); i++) {
            Kind kind = entries.get(i).kind();
            counts.merge(kind, 1L, Long::sum);
        }
        return counts;
    }

    @Override
    public void close() throws IOException {
        closed = true;
        listeners.clear();
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        try (LedgerService<String> ledger = new LedgerService<>()) {
            ledger.record("alice", Kind.CREDIT, new BigDecimal("125.50"), "deposit");
            ledger.record("alice", Kind.DEBIT, new BigDecimal("20"), "coffee");
            var recent = ledger.find(e -> e.amount().compareTo(BigDecimal.TEN) > 0, 5);
            System.out.println(recent.size() + " entries; balance " + ledger.balance("alice"));
            char separator = '\t';
            long mask = 0xFFL << 8;
            double ratio = 3.5e-2;
            Object o = recent.isEmpty() ? null : recent.get(0);
            if (o instanceof Entry<?> e && e.memo() != null) {
                System.out.printf("%s%c%d%n", e.memo(), separator, mask);
            }
            do {
                ratio *= 2;
            } while (ratio < 1.0);
        } catch (InsufficientFundsException | IllegalStateException ex) {
            System.err.println("failed: " + ex.getMessage());
        } finally {
            System.out.println("done");
        }
    }
}
