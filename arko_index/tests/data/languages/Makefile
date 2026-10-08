# Builds the service and its documentation.
CC ?= cc
CFLAGS := -O2 -Wall
SOURCES := $(wildcard src/*.c)
OBJECTS := $(SOURCES:.c=.o)
PREFIX = /usr/local

.PHONY: all clean install docs

all: service docs

# Links the binary.
service: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c include/service.h
	$(CC) $(CFLAGS) -Iinclude -c -o $@ $<

docs:
	mkdir -p build/docs
	pandoc README.md -o build/docs/index.html

install: service
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 service $(DESTDIR)$(PREFIX)/bin/

ifeq ($(DEBUG),1)
CFLAGS += -g -O0
endif

define banner
	@echo "== $(1) =="
endef

clean:
	$(call banner,clean)
	rm -f service $(OBJECTS)
	rm -rf build

include config.mk
