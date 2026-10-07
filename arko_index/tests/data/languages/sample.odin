package geometry

import "core:fmt"
import "core:math"

// A point in the plane.
Point :: struct {
	x, y: f64,
}

Shape :: union {
	Circle,
	Rect,
}

Circle :: struct {
	center: Point,
	radius: f64,
}

Rect :: struct {
	min, max: Point,
}

Color :: enum {
	Red,
	Green,
	Blue,
}

// The distance between two points.
distance :: proc(a, b: Point) -> f64 {
	dx := a.x - b.x
	dy := a.y - b.y
	return math.sqrt(dx * dx + dy * dy)
}

area :: proc(s: Shape) -> f64 {
	switch v in s {
	case Circle:
		return math.PI * v.radius * v.radius
	case Rect:
		return (v.max.x - v.min.x) * (v.max.y - v.min.y)
	}
	return 0
}

ORIGIN :: Point{0, 0}

main :: proc() {
	fmt.println(distance(ORIGIN, Point{3, 4}))
	fmt.println(area(Circle{ORIGIN, 1}))
}
