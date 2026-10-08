package shop.orders

import scala.collection.mutable

/** An order line. */
final case class OrderLine(sku: String, quantity: Int)

sealed trait Status
object Status {
  case object Open extends Status
  case object Paid extends Status
}

trait Pricing {
  def priceOf(sku: String): BigDecimal
}

/** Totals orders. */
class OrderTotals(pricing: Pricing) {
  private val cache = mutable.Map.empty[String, BigDecimal]

  def total(lines: Seq[OrderLine]): BigDecimal =
    lines.map(l => price(l.sku) * l.quantity).sum

  // Cached lookup.
  private def price(sku: String): BigDecimal =
    cache.getOrElseUpdate(sku, pricing.priceOf(sku))
}

enum Color:
  case Red, Green, Blue

object Main {
  val zero: Pricing = _ => BigDecimal(0)

  def main(args: Array[String]): Unit = {
    val totals = new OrderTotals(zero)
    println(totals.total(Seq(OrderLine("apple", 2))))
  }
}

given Ordering[OrderLine] = Ordering.by(_.sku)

extension (l: OrderLine) def doubled: OrderLine = l.copy(quantity = l.quantity * 2)

type Lines = Seq[OrderLine]
