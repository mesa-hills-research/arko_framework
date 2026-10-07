<?php

declare(strict_types=1);

namespace Shop\Orders;

use Shop\Pricing\PricingInterface;

/**
 * Totals orders.
 */
#[Attribute]
final class OrderTotals
{
    private array $cache = [];

    public function __construct(private PricingInterface $pricing)
    {
    }

    /** The total of the order lines. */
    public function total(array $lines): float
    {
        $total = 0.0;
        foreach ($lines as $line) {
            $total += $this->price($line['sku']) * $line['qty'];
        }
        return $total;
    }

    private function price(string $sku): float
    {
        return $this->cache[$sku] ??= $this->pricing->priceOf($sku);
    }
}

interface Describable
{
    public function describe(): string;
}

trait Loggable
{
    public function log(string $message): void
    {
        error_log(static::class . ': ' . $message);
    }
}

enum Status: string
{
    case Open = 'open';
    case Paid = 'paid';
}

function format_money(float $amount): string
{
    return number_format($amount, 2);
}
?>
<p>Total: <?= format_money(12.5) ?></p>
