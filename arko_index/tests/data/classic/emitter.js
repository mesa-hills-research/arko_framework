// A tiny event emitter.
"use strict";

const DEFAULT_LIMIT = 10;

class Emitter {
  constructor(limit = DEFAULT_LIMIT) {
    this.handlers = new Map();
    this.limit = limit;
  }

  /** Registers a handler. */
  on(event, handler) {
    const list = this.handlers.get(event) ?? [];
    if (list.length >= this.limit) throw new Error("too many handlers");
    list.push(handler);
    this.handlers.set(event, list);
    return () => this.off(event, handler);
  }

  off(event, handler) {
    const list = this.handlers.get(event) ?? [];
    this.handlers.set(event, list.filter((h) => h !== handler));
  }

  emit(event, ...args) {
    for (const handler of this.handlers.get(event) ?? []) handler(...args);
  }
}

const helpers = {
  count: (emitter, event) => (emitter.handlers.get(event) ?? []).length,
  clear: function (emitter) {
    emitter.handlers.clear();
  },
};

export function once(emitter, event, handler) {
  const off = emitter.on(event, (...args) => {
    off();
    handler(...args);
  });
  return off;
}

export const debounce = (fn, ms) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

module.exports = { Emitter, helpers };
