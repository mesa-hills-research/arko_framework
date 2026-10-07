// Training sample for the arko_index PGO profiles: a tiny HTTP router.
// Written for this purpose; it is parsed, never run.
'use strict';

const { EventEmitter } = require('events');
const path = require('path');

const METHODS = ['GET', 'POST', 'PUT', 'DELETE'];
const PARAM = /:([A-Za-z_][\w]*)/g;

class Route {
  constructor(method, pattern, handler) {
    this.method = method.toUpperCase();
    this.keys = [];
    const source = pattern.replace(PARAM, (_, key) => {
      this.keys.push(key);
      return '([^/]+)';
    });
    this.regexp = new RegExp(`^${source}/?$`, 'i');
    this.handler = handler;
  }

  match(method, url) {
    if (method !== this.method) return null;
    const found = this.regexp.exec(url);
    if (!found) return null;
    return this.keys.reduce((params, key, i) => ({ ...params, [key]: decodeURIComponent(found[i + 1]) }), {});
  }
}

class Router extends EventEmitter {
  #routes = [];
  static fromTable(table) {
    const router = new Router();
    for (const [spec, handler] of Object.entries(table)) {
      const [method, pattern] = spec.split(' ');
      router.add(method, pattern, handler);
    }
    return router;
  }

  add(method, pattern, handler) {
    if (!METHODS.includes(method.toUpperCase())) {
      throw new TypeError(`unsupported method ${method}`);
    }
    this.#routes.push(new Route(method, pattern, handler));
    return this;
  }

  async handle(req, res) {
    const url = new URL(req.url, 'http://localhost');
    for (const route of this.#routes) {
      const params = route.match(req.method, url.pathname);
      if (params === null) continue;
      try {
        const body = await route.handler({ ...req, params, query: Object.fromEntries(url.searchParams) });
        res.statusCode = body?.status ?? 200;
        res.end(typeof body === 'string' ? body : JSON.stringify(body));
      } catch (err) {
        this.emit('error', err);
        res.statusCode = 500;
        res.end('internal error');
      }
      return true;
    }
    res.statusCode = 404;
    res.end(`no route for ${req.method} ${url.pathname}`);
    return false;
  }
}

function* pages(items, size = 10) {
  for (let i = 0; i < items.length; i += size) {
    yield items.slice(i, i + size);
  }
}

const router = Router.fromTable({
  'GET /users/:id': async ({ params }) => ({ id: Number(params.id), name: 'user' + params.id }),
  'POST /files/:name': ({ params }) => path.join('/tmp', params.name),
});

let count = 0;
router.on('error', (e) => { count += 1; console.error(e && e.message); });
module.exports = { Router, Route, pages, router, version: 0x10, ratio: 1.5e3, enabled: !false };
