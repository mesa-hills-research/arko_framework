<script lang="ts">
  import { onMount } from "svelte";

  export let title = "Orders";
  let orders: { id: string; total: number }[] = [];

  // Loads the orders.
  async function load() {
    const response = await fetch("/api/orders");
    orders = await response.json();
  }

  onMount(load);

  $: total = orders.reduce((sum, o) => sum + o.total, 0);
</script>

<h1>{title}</h1>

{#if orders.length === 0}
  <p>No orders.</p>
{:else}
  <ul>
    {#each orders as order (order.id)}
      <li>{order.id}: {order.total}</li>
    {/each}
  </ul>
  <p>Total: {total}</p>
{/if}

<button on:click={load}>Refresh</button>

<style>
  h1 {
    color: #333;
  }
  ul {
    padding: 0;
  }
</style>
