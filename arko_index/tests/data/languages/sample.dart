import 'dart:async';
import 'package:meta/meta.dart';

/// A point in the plane.
@immutable
class Point {
  final double x;
  final double y;

  const Point(this.x, this.y);

  /// The distance to [other].
  double distanceTo(Point other) {
    final dx = x - other.x;
    final dy = y - other.y;
    return (dx * dx + dy * dy).sqrt();
  }

  Point operator +(Point other) => Point(x + other.x, y + other.y);

  @override
  String toString() => 'Point($x, $y)';
}

mixin Describable {
  String describe() => runtimeType.toString();
}

enum Shape { circle, square, triangle }

extension PointList on List<Point> {
  Point get centroid {
    var sx = 0.0, sy = 0.0;
    for (final p in this) {
      sx += p.x;
      sy += p.y;
    }
    return Point(sx / length, sy / length);
  }
}

typedef Transform = Point Function(Point);

Future<void> main() async {
  final points = [const Point(0, 0), const Point(3, 4)];
  print(points.centroid);
  await Future.delayed(const Duration(milliseconds: 10));
}
