#ifndef RING_H
#define RING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ring;

/* Creates a ring of `size` bytes; NULL when out of memory. */
struct ring *ring_new(size_t size);
int ring_push(struct ring *r, unsigned char byte);
int ring_pop(struct ring *r, unsigned char *byte);
void ring_free(struct ring *r);

#ifdef __cplusplus
}

namespace ring {
template <typename T>
class Ring {
  public:
    explicit Ring(size_t size) : size_(size) {}
    size_t size() const { return size_; }

  private:
    size_t size_;
};
}  // namespace ring
#endif

#endif
