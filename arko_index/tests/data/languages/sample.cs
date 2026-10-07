using System;
using System.Collections.Generic;

namespace Shop.Orders
{
    /// <summary>An order line: a product and a quantity.</summary>
    public record OrderLine(string Sku, int Quantity);

    public interface IPricing
    {
        decimal PriceOf(string sku);
    }

    /// <summary>Totals orders with a pricing source.</summary>
    [Serializable]
    public class OrderTotals : IDisposable
    {
        private readonly IPricing _pricing;
        private readonly Dictionary<string, decimal> _cache = new();

        public OrderTotals(IPricing pricing)
        {
            _pricing = pricing;
        }

        public decimal Total(IEnumerable<OrderLine> lines)
        {
            decimal total = 0;
            foreach (var line in lines)
            {
                total += Price(line.Sku) * line.Quantity;
            }
            return total;
        }

        // Cached price lookup.
        private decimal Price(string sku)
        {
            if (!_cache.TryGetValue(sku, out var price))
            {
                price = _pricing.PriceOf(sku);
                _cache[sku] = price;
            }
            return price;
        }

        public string Name { get; set; } = "totals";

        public void Dispose() => _cache.Clear();
    }

    public enum Status { Open, Paid, Shipped }

    public struct Money
    {
        public decimal Amount;
        public string Currency;
    }
}
