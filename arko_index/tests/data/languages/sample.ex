defmodule Shop.Cart do
  @moduledoc """
  A shopping cart: items and their quantities.
  """

  alias Shop.{Item, Pricing}
  require Logger

  @type t :: %__MODULE__{items: map()}
  defstruct items: %{}

  @doc "Adds `qty` of `item` to the cart."
  @spec add(t(), Item.t(), pos_integer()) :: t()
  def add(%__MODULE__{items: items} = cart, item, qty \\ 1) do
    Logger.debug("adding #{item.sku}")
    %{cart | items: Map.update(items, item.sku, qty, &(&1 + qty))}
  end

  @doc "The cart's total price."
  def total(%__MODULE__{items: items}) do
    items
    |> Enum.map(fn {sku, qty} -> Pricing.price(sku) * qty end)
    |> Enum.sum()
  end

  defp empty?(%__MODULE__{items: items}), do: map_size(items) == 0

  defmacro with_cart(do: block) do
    quote do
      cart = %Shop.Cart{}
      unquote(block)
    end
  end

  defprotocol Describe do
    def describe(value)
  end
end

defmodule Shop.Item do
  defstruct [:sku, :name]
end
