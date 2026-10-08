#import <Foundation/Foundation.h>
#import "Shape.h"

// A rectangle shape.
@interface Rectangle : NSObject <Shape>

@property (nonatomic, assign) double width;
@property (nonatomic, assign) double height;

- (instancetype)initWithWidth:(double)width height:(double)height;
- (double)area;
+ (instancetype)square:(double)side;

@end

@implementation Rectangle

- (instancetype)initWithWidth:(double)width height:(double)height {
    self = [super init];
    if (self) {
        _width = width;
        _height = height;
    }
    return self;
}

/// The area.
- (double)area {
    return self.width * self.height;
}

+ (instancetype)square:(double)side {
    return [[self alloc] initWithWidth:side height:side];
}

@end

@protocol Drawable <NSObject>
- (void)drawInContext:(id)context;
@end

static double totalArea(NSArray<Rectangle *> *shapes) {
    double sum = 0;
    for (Rectangle *r in shapes) {
        sum += [r area];
    }
    return sum;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSLog(@"%f", totalArea(@[[Rectangle square:2]]));
    }
    return 0;
}
