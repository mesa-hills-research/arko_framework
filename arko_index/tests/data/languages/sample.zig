const std = @import("std");

/// A point in the plane.
pub const Point = struct {
    x: f64,
    y: f64,

    /// The distance to another point.
    pub fn distance(self: Point, other: Point) f64 {
        const dx = self.x - other.x;
        const dy = self.y - other.y;
        return @sqrt(dx * dx + dy * dy);
    }

    pub fn add(self: Point, other: Point) Point {
        return .{ .x = self.x + other.x, .y = self.y + other.y };
    }
};

pub const Shape = union(enum) {
    circle: f64,
    rect: struct { w: f64, h: f64 },
};

const Color = enum { red, green, blue };

// The area of a shape.
fn area(s: Shape) f64 {
    return switch (s) {
        .circle => |r| std.math.pi * r * r,
        .rect => |r| r.w * r.h,
    };
}

test "distance" {
    const a = Point{ .x = 0, .y = 0 };
    const b = Point{ .x = 3, .y = 4 };
    try std.testing.expectEqual(@as(f64, 5), a.distance(b));
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("{d}\n", .{area(.{ .circle = 1 })});
}
