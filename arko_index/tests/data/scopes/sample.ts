// Scope fixture: a namespace, a class large enough to expose its methods, an
// interface, an exported function and an arrow function bound to a name.
import { EventEmitter } from "events";

export namespace Storage {
  export interface Options {
    directory: string;
    durable: boolean;
  }
  export class Journal {
    entries: string[] = [];
    append(entry: string): void {
      this.entries.push(entry);
    }
  }
}

export class ConnectionPool extends EventEmitter {
  private readonly idle: string[] = [];
  private readonly busy = new Set<string>();

  constructor(private readonly capacity: number) {
    super();
  }

  acquire(): string | undefined {
    const connection = this.idle.pop();
    if (connection !== undefined) {
      this.busy.add(connection);
      this.emit("acquire", connection);
      return connection;
    }
    if (this.busy.size < this.capacity) {
      const fresh = `connection-${this.busy.size + 1}`;
      this.busy.add(fresh);
      this.emit("create", fresh);
      return fresh;
    }
    return undefined;
  }

  release(connection: string): void {
    if (!this.busy.delete(connection)) {
      throw new Error(`unknown connection ${connection}`);
    }
    this.idle.push(connection);
    this.emit("release", connection);
  }

  get size(): number {
    return this.idle.length + this.busy.size;
  }
}

export function retryDelay(attempt: number, base = 100): number {
  return Math.min(base * 2 ** attempt, 30_000);
}

const formatBytes = (bytes: number): string => {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
};

export default formatBytes;
