(* A persistent stack and a queue built from two stacks. *)

module Stack = struct
  type 'a t = 'a list

  let empty = []

  (** Pushes [x]. *)
  let push x s = x :: s

  let pop = function
    | [] -> None
    | x :: rest -> Some (x, rest)
end

module type QUEUE = sig
  type 'a t
  val empty : 'a t
  val enqueue : 'a -> 'a t -> 'a t
  val dequeue : 'a t -> ('a * 'a t) option
end

module Queue : QUEUE = struct
  type 'a t = { front : 'a list; back : 'a list }

  let empty = { front = []; back = [] }

  let enqueue x q = { q with back = x :: q.back }

  let rec dequeue q =
    match q.front with
    | x :: front -> Some (x, { q with front })
    | [] when q.back = [] -> None
    | [] -> dequeue { front = List.rev q.back; back = [] }
end

type shape =
  | Circle of float
  | Rect of float * float

exception Empty

let area = function
  | Circle r -> 3.14159 *. r *. r
  | Rect (w, h) -> w *. h

class counter = object
  val mutable n = 0
  method incr = n <- n + 1
  method get = n
end

let () =
  let q = Queue.enqueue 1 Queue.empty in
  match Queue.dequeue q with
  | Some (x, _) -> print_int x
  | None -> raise Empty
