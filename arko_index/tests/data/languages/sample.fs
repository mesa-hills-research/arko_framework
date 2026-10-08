namespace Shop

open System
open System.Collections.Generic

/// An order line.
type OrderLine = { Sku: string; Quantity: int }

type Status =
    | Open
    | Paid of DateTime
    | Shipped of carrier: string

/// Prices orders.
type Pricing(prices: IDictionary<string, decimal>) =
    member _.PriceOf(sku: string) =
        match prices.TryGetValue sku with
        | true, p -> p
        | _ -> 0m

    member this.Total(lines: OrderLine list) =
        lines |> List.sumBy (fun l -> this.PriceOf l.Sku * decimal l.Quantity)

    interface IDisposable with
        member _.Dispose() = ()

module Orders =
    // The empty order.
    let empty: OrderLine list = []

    let add sku qty lines = { Sku = sku; Quantity = qty } :: lines

    let rec count lines =
        match lines with
        | [] -> 0
        | l :: rest -> l.Quantity + count rest

    [<EntryPoint>]
    let main argv =
        let lines = empty |> add "apple" 2 |> add "pear" 1
        printfn "%d items" (count lines)
        0
