//! Scope fixture: modules, an inherent impl, a trait impl, a trait and free functions.
use std::fmt;

pub mod geometry {
    pub mod shapes {
        use super::super::fmt;

        /// A point in the plane.
        pub struct Point {
            pub x: f64,
            pub y: f64,
        }

        impl Point {
            /// The distance to another point.
            pub fn distance(&self, other: &Point) -> f64 {
                let dx = self.x - other.x;
                let dy = self.y - other.y;
                (dx * dx + dy * dy).sqrt()
            }

            pub fn origin() -> Self {
                Point { x: 0.0, y: 0.0 }
            }
        }

        impl fmt::Display for Point {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "({}, {})", self.x, self.y)
            }
        }
    }
}

pub trait Area {
    fn area(&self) -> f64;
    fn describe(&self) -> String {
        format!("area {:.2}", self.area())
    }
}

impl<T: Area> Area for Vec<T> {
    fn area(&self) -> f64 {
        self.iter().map(Area::area).sum()
    }
}

macro_rules! square {
    ($x:expr) => {
        $x * $x
    };
}

pub fn exponential_backoff(attempt: u32) -> u64 {
    let base: u64 = 100;
    base.saturating_mul(square!(u64::from(attempt)))
}
