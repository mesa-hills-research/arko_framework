import Foundation

/// An order line.
struct OrderLine: Codable {
    let sku: String
    let quantity: Int
}

protocol Pricing {
    func price(of sku: String) -> Decimal
}

enum Status: String {
    case open, paid, shipped
}

/// Totals orders.
final class OrderTotals {
    private let pricing: Pricing
    private var cache: [String: Decimal] = [:]

    init(pricing: Pricing) {
        self.pricing = pricing
    }

    func total(_ lines: [OrderLine]) -> Decimal {
        lines.reduce(0) { $0 + price($1.sku) * Decimal($1.quantity) }
    }

    // Cached lookup.
    private func price(_ sku: String) -> Decimal {
        if let cached = cache[sku] { return cached }
        let p = pricing.price(of: sku)
        cache[sku] = p
        return p
    }

    var count: Int { cache.count }

    deinit {
        cache.removeAll()
    }
}

extension OrderLine: CustomStringConvertible {
    var description: String { "\(sku) x\(quantity)" }
}

@MainActor
func run() async throws {
    let totals = OrderTotals(pricing: FixedPricing())
    print(totals.total([OrderLine(sku: "apple", quantity: 2)]))
}

struct FixedPricing: Pricing {
    func price(of sku: String) -> Decimal { 1 }
}
