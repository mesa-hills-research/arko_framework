package shop.orders

import java.math.BigDecimal

/** An order line. */
data class OrderLine(val sku: String, val quantity: Int)

interface Pricing {
    fun priceOf(sku: String): BigDecimal
}

enum class Status { OPEN, PAID, SHIPPED }

/** Totals orders. */
class OrderTotals(private val pricing: Pricing) {
    private val cache = mutableMapOf<String, BigDecimal>()

    fun total(lines: List<OrderLine>): BigDecimal =
        lines.fold(BigDecimal.ZERO) { acc, line ->
            acc + price(line.sku) * line.quantity.toBigDecimal()
        }

    // Cached lookup.
    private fun price(sku: String): BigDecimal =
        cache.getOrPut(sku) { pricing.priceOf(sku) }

    companion object {
        const val CURRENCY = "USD"

        fun empty() = OrderTotals(object : Pricing {
            override fun priceOf(sku: String) = BigDecimal.ZERO
        })
    }
}

object Registry {
    val totals = mutableListOf<OrderTotals>()
}

@Suppress("unused")
fun main() {
    val totals = OrderTotals.empty()
    println(totals.total(listOf(OrderLine("apple", 2))))
}

typealias Lines = List<OrderLine>
