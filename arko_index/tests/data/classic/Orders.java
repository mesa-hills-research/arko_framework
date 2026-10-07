package shop.orders;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** An order line. */
record OrderLine(String sku, int quantity) {}

interface Pricing {
    BigDecimal priceOf(String sku);
}

enum Status { OPEN, PAID, SHIPPED }

/** Totals orders. */
@SuppressWarnings("unused")
public final class Orders {
    private final Pricing pricing;
    private final Map<String, BigDecimal> cache = new HashMap<>();

    public Orders(Pricing pricing) {
        this.pricing = pricing;
    }

    /** The total of the lines. */
    public BigDecimal total(List<OrderLine> lines) {
        BigDecimal sum = BigDecimal.ZERO;
        for (OrderLine line : lines) {
            sum = sum.add(price(line.sku()).multiply(BigDecimal.valueOf(line.quantity())));
        }
        return sum;
    }

    // Cached lookup.
    private BigDecimal price(String sku) {
        return cache.computeIfAbsent(sku, pricing::priceOf);
    }

    public int getCount() { return cache.size(); }
    public void setCount(int count) { }

    static class Builder {
        private Pricing pricing;

        Builder pricing(Pricing pricing) {
            this.pricing = pricing;
            return this;
        }

        Orders build() {
            return new Orders(pricing);
        }
    }
}
