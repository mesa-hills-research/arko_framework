---
title: Geometry guide
tags: [math, guide]
---

# Geometry

A small library for points and polygons.

## Install

Add the crate:

```sh
cargo add geometry
```

### Linux

Nothing else is needed.

### Windows

Install the build tools first.

## Usage

Create points and measure them:

```rust
use geometry::Point;

fn main() {
    let a = Point::new(0.0, 0.0);
    let b = Point::new(3.0, 4.0);
    println!("{}", a.distance(&b));
}
```

- Points are `Copy`.
- Polygons own their points.

> Areas are always positive.

| Shape | Area |
|---|---|
| Unit square | 1 |

Setext heading
--------------

Text under a setext heading.

<details>
<summary>More</summary>

Hidden text.
</details>
