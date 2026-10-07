/* Training sample for the arko_index PGO profiles: a ring buffer and a line reader.
 * Written for this purpose; it is parsed, never compiled. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RING_MIN_CAPACITY 64u
#define RING_ALIGN(n) (((n) + 7u) & ~7u)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#if defined(__GNUC__)
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

typedef struct ring {
    uint8_t *data;
    size_t capacity;
    size_t head; /* next byte to read */
    size_t len;  /* bytes stored */
} ring_t;

enum ring_status { RING_OK = 0, RING_FULL = -1, RING_EMPTY = -2, RING_NOMEM = -3 };

static const char *const status_names[] = {
    [0] = "ok", [1] = "full", [2] = "empty", [3] = "out of memory",
};

int ring_init(ring_t *r, size_t capacity) {
    if (capacity < RING_MIN_CAPACITY) capacity = RING_MIN_CAPACITY;
    r->data = malloc(RING_ALIGN(capacity));
    if (r->data == NULL) return RING_NOMEM;
    r->capacity = capacity;
    r->head = r->len = 0;
    return RING_OK;
}

void ring_free(ring_t *r) {
    free(r->data);
    memset(r, 0, sizeof *r);
}

size_t ring_write(ring_t *r, const void *src, size_t n) {
    const uint8_t *p = (const uint8_t *)src;
    size_t written = 0;
    while (written < n && r->len < r->capacity) {
        size_t tail = (r->head + r->len) % r->capacity;
        size_t room = MIN(r->capacity - r->len, r->capacity - tail);
        size_t chunk = MIN(room, n - written);
        memcpy(r->data + tail, p + written, chunk);
        r->len += chunk;
        written += chunk;
    }
    return written;
}

size_t ring_read(ring_t *r, void *dst, size_t n) {
    uint8_t *out = dst;
    size_t got = 0;
    for (; got < n && r->len > 0; ++got) {
        out[got] = r->data[r->head];
        r->head = (r->head + 1) % r->capacity;
        r->len--;
    }
    return got;
}

static int read_line(ring_t *r, char *line, size_t max) {
    size_t i = 0;
    while (i + 1 < max) {
        char c;
        if (ring_read(r, &c, 1) == 0) return i > 0 ? (int)i : RING_EMPTY;
        if (c == '\n') break;
        line[i++] = c;
    }
    line[i] = '\0';
    return (int)i;
}

static const char *status_name(int status) {
    switch (status) {
    case RING_OK:
        return status_names[0];
    case RING_FULL:
        return status_names[1];
    case RING_EMPTY:
        return status_names[2];
    default:
        return status_names[3];
    }
}

int main(int argc, char **argv UNUSED) {
    ring_t ring;
    char line[128];
    unsigned long total = 0UL;
    double average = 0.0;
    if (ring_init(&ring, 256) != RING_OK) {
        fprintf(stderr, "init failed: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    const char *text = "alpha\nbeta\ngamma\n";
    ring_write(&ring, text, strlen(text));
    int n;
    while ((n = read_line(&ring, line, sizeof line)) > 0) {
        total += (unsigned long)n;
        printf("%3d: %s\n", n, line);
    }
    average = total / 3.0;
    printf("status %s, total %lu, average %.2f, argc %d, hex %#x\n", status_name(n), total, average, argc, 0xBEEFu);
    ring_free(&ring);
    return 0;
}
