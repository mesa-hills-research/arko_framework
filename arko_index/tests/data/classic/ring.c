/* A fixed-size ring buffer. */
#include <stdlib.h>
#include <string.h>
#include "ring.h"

#define RING_MIN 4

struct ring {
    size_t head, tail, size;
    unsigned char *data;
};

static size_t next(const struct ring *r, size_t i) {
    return (i + 1) % r->size;
}

/* Creates a ring of `size` bytes. */
struct ring *ring_new(size_t size) {
    struct ring *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->size = size < RING_MIN ? RING_MIN : size;
    r->data = malloc(r->size);
    if (!r->data) {
        free(r);
        return NULL;
    }
    return r;
}

int ring_push(struct ring *r, unsigned char byte) {
    if (next(r, r->head) == r->tail) return -1;
    r->data[r->head] = byte;
    r->head = next(r, r->head);
    return 0;
}

int ring_pop(struct ring *r, unsigned char *byte) {
    if (r->head == r->tail) return -1;
    *byte = r->data[r->tail];
    r->tail = next(r, r->tail);
    return 0;
}

void ring_free(struct ring *r) {
    if (r) free(r->data);
    free(r);
}
