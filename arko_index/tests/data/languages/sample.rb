# frozen_string_literal: true

require "json"

module Shop
  # A shopping cart.
  class Cart
    include Enumerable

    attr_reader :items

    def initialize(currency: "USD")
      @items = Hash.new(0)
      @currency = currency
    end

    # Adds +qty+ of +sku+.
    def add(sku, qty = 1)
      @items[sku] += qty
      self
    end

    def each(&block)
      @items.each(&block)
    end

    def total(prices)
      sum { |sku, qty| prices.fetch(sku, 0) * qty }
    end

    def self.from_json(text)
      data = JSON.parse(text)
      new(currency: data["currency"]).tap do |cart|
        data["items"].each { |sku, qty| cart.add(sku, qty) }
      end
    end

    private

    def empty?
      @items.empty?
    end
  end

  module Pricing
    PRICES = { "apple" => 0.5 }.freeze

    def self.price(sku) = PRICES.fetch(sku)
  end
end

cart = Shop::Cart.new.add("apple", 2)
puts cart.total(Shop::Pricing::PRICES)
