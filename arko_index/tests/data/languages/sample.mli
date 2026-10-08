(** Persistent queues. *)

type 'a t
(** A queue of ['a]. *)

val empty : 'a t

(** [enqueue x q] adds [x] at the back. *)
val enqueue : 'a -> 'a t -> 'a t

val dequeue : 'a t -> ('a * 'a t) option

exception Empty

module type ORDERED = sig
  type t
  val compare : t -> t -> int
end

module Make (O : ORDERED) : sig
  type elt = O.t
  type t
  val add : elt -> t -> t
  val min : t -> elt option
end

class type counter = object
  method incr : unit
  method get : int
end

external length : 'a array -> int = "%array_length"
