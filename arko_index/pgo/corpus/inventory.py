"""Training sample for the arko_index PGO profiles: a small inventory service.

Written for this purpose; it is parsed, never run.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from collections import defaultdict
from typing import Any, Callable, Iterable, Optional

VERSION = (1, 4, 2)
DEFAULT_LIMIT = 0x40
RATE = 1.5e-3
PATTERN = re.compile(r"^(?P<sku>[A-Z]{3}-\d{4})(?:/(?P<lot>\w+))?$")


class InventoryError(Exception):
    """Raised when a reservation cannot be satisfied."""

    def __init__(self, sku: str, wanted: int, available: int) -> None:
        super().__init__(f"{sku}: wanted {wanted}, only {available} available")
        self.sku = sku
        self.wanted = wanted
        self.available = available


@dataclasses.dataclass(frozen=True)
class Item:
    sku: str
    name: str
    price_cents: int
    tags: tuple[str, ...] = ()

    @property
    def price(self) -> float:
        return self.price_cents / 100

    def matches(self, query: str) -> bool:
        q = query.lower()
        return q in self.name.lower() or any(q == t for t in self.tags)


class Inventory:
    def __init__(self, items: Iterable[Item] = (), *, limit: int = DEFAULT_LIMIT):
        self._items = {item.sku: item for item in items}
        self._stock: dict[str, int] = defaultdict(int)
        self._listeners: list[Callable[[str, int], None]] = []
        self.limit = limit

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, sku: object) -> bool:
        return sku in self._items

    def add(self, item: Item, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self._items.setdefault(item.sku, item)
        self._stock[item.sku] += count
        for listener in self._listeners:
            listener(item.sku, self._stock[item.sku])

    def reserve(self, sku: str, count: int) -> int:
        available = self._stock.get(sku, 0)
        if count > available:
            raise InventoryError(sku, count, available)
        self._stock[sku] = available - count
        return self._stock[sku]

    def search(self, query: str, limit: Optional[int] = None) -> list[Item]:
        found = [item for item in self._items.values() if item.matches(query)]
        found.sort(key=lambda item: (-self._stock[item.sku], item.name))
        return found[: limit or self.limit]

    def on_change(self, listener: Callable[[str, int], None]) -> None:
        self._listeners.append(listener)

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "version": ".".join(map(str, VERSION)),
            "items": [dataclasses.asdict(i) for i in self._items.values()],
            "stock": dict(self._stock),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Inventory":
        data = json.loads(text)
        inventory = cls(Item(**raw) for raw in data.get("items", []))
        for sku, count in data.get("stock", {}).items():
            if count:
                inventory._stock[sku] = int(count)
        return inventory


def parse_sku(text: str) -> tuple[str, Optional[str]]:
    match = PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"not a SKU: {text!r}")
    return match["sku"], match.group("lot")


async def restock(inventory: Inventory, deliveries: list[tuple[Item, int]], delay: float = RATE):
    for item, count in deliveries:
        await asyncio.sleep(delay)
        try:
            inventory.add(item, count)
        except ValueError as error:
            print(f"skipped {item.sku}: {error}")
        else:
            yield item.sku
        finally:
            pass


def summarize(inventory: Inventory) -> str:
    lines = []
    total = 0
    for sku in sorted(inventory._items):
        count = inventory._stock[sku]
        total += count * inventory._items[sku].price_cents
        status = "out" if count == 0 else ("low" if count < 5 else "ok")
        lines.append(f"{sku:<10} {count:>5} {status}")
    lines.append(f"total value: {total / 100:,.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    shop = Inventory([Item("ABC-0001", "Widget", 1999, ("tools",)), Item("XYZ-0042", "Gadget", 4999)])
    shop.add(shop.search("widget")[0], 3)
    with open("/dev/null", "w") as sink:
        print(summarize(shop), file=sink)
    while len(shop) > 10:
        break
    squares = {n: n**2 for n in range(10) if n % 2 == 0}
    assert all(v >= 0 for v in squares.values()), "squares are non-negative"
