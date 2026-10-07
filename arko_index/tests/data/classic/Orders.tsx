import React, { useEffect, useState } from "react";

/** An order as the API returns it. */
export interface Order {
  id: string;
  total: number;
}

type Props = { title: string; refreshMs?: number };

async function fetchOrders(): Promise<Order[]> {
  const response = await fetch("/api/orders");
  return response.json();
}

export function OrderList({ title, refreshMs = 5000 }: Props) {
  const [orders, setOrders] = useState<Order[]>([]);

  useEffect(() => {
    const load = () => fetchOrders().then(setOrders);
    load();
    const timer = setInterval(load, refreshMs);
    return () => clearInterval(timer);
  }, [refreshMs]);

  const total = orders.reduce((sum, o) => sum + o.total, 0);

  return (
    <section>
      <h1>{title}</h1>
      <ul>
        {orders.map((o) => (
          <li key={o.id}>
            {o.id}: {o.total}
          </li>
        ))}
      </ul>
      <p>Total: {total}</p>
    </section>
  );
}

export default class OrderPage extends React.Component<Props> {
  render() {
    return <OrderList title={this.props.title} />;
  }
}
