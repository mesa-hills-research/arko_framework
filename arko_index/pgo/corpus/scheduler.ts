// Training sample for the arko_index PGO profiles: a small job scheduler.
// Written for this purpose; it is parsed, never run.
import { EventEmitter } from "events";
import type { Readable } from "stream";

export type JobId = string & { readonly __brand: "JobId" };
export type Priority = "low" | "normal" | "high";

export interface JobSpec<T = unknown> {
  id: JobId;
  priority?: Priority;
  retries: number;
  run(input: T, signal: AbortSignal): Promise<T>;
}

export enum JobState {
  Queued = "queued",
  Running = "running",
  Done = "done",
  Failed = "failed",
}

const PRIORITY_WEIGHT: Record<Priority, number> = { low: 1, normal: 4, high: 16 };
const MAX_BACKOFF_MS = 30_000;

/** Exponential backoff with full jitter. */
export function backoff(attempt: number, baseMs = 250, random: () => number = Math.random): number {
  const cap = Math.min(MAX_BACKOFF_MS, baseMs * 2 ** attempt);
  return Math.floor(random() * cap);
}

abstract class Store<K, V> {
  protected readonly entries = new Map<K, V>();
  abstract key(value: V): K;
  put(value: V): this {
    this.entries.set(this.key(value), value);
    return this;
  }
  get size(): number {
    return this.entries.size;
  }
}

class JobStore extends Store<JobId, { spec: JobSpec; state: JobState; attempts: number }> {
  key(value: { spec: JobSpec }): JobId {
    return value.spec.id;
  }
  byState(state: JobState) {
    return [...this.entries.values()].filter((entry) => entry.state === state);
  }
}

export class Scheduler extends EventEmitter {
  private readonly store = new JobStore();
  private running = 0;
  #paused = false;

  constructor(private readonly concurrency: number = 4, public name?: string) {
    super();
    if (concurrency < 1 || !Number.isInteger(concurrency)) {
      throw new RangeError(`concurrency must be a positive integer, got ${concurrency}`);
    }
  }

  submit<T>(spec: JobSpec<T>): JobId {
    this.store.put({ spec: spec as JobSpec, state: JobState.Queued, attempts: 0 });
    this.emit("queued", spec.id);
    void this.pump();
    return spec.id;
  }

  pause(): void {
    this.#paused = true;
  }

  resume(): void {
    this.#paused = false;
    void this.pump();
  }

  private next() {
    const queued = this.store.byState(JobState.Queued);
    queued.sort((a, b) => PRIORITY_WEIGHT[b.spec.priority ?? "normal"] - PRIORITY_WEIGHT[a.spec.priority ?? "normal"]);
    return queued[0];
  }

  private async pump(): Promise<void> {
    while (!this.#paused && this.running < this.concurrency) {
      const entry = this.next();
      if (!entry) break;
      entry.state = JobState.Running;
      this.running++;
      const controller = new AbortController();
      try {
        await entry.spec.run(undefined, controller.signal);
        entry.state = JobState.Done;
        this.emit("done", entry.spec.id);
      } catch (error: unknown) {
        entry.attempts += 1;
        const message = error instanceof Error ? error.message : String(error);
        if (entry.attempts <= entry.spec.retries) {
          entry.state = JobState.Queued;
          await new Promise((resolve) => setTimeout(resolve, backoff(entry.attempts)));
        } else {
          entry.state = JobState.Failed;
          this.emit("failed", entry.spec.id, message);
        }
      } finally {
        this.running--;
      }
    }
  }
}

export async function drain(stream: Readable): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : (chunk as Buffer));
  }
  return Buffer.concat(chunks).toString("utf8");
}

export const summary = (s: Scheduler): string => `${s.name ?? "scheduler"}: ${JSON.stringify({ ok: true })}`;

declare module "events" {
  interface EventEmitter {
    tag?: string;
  }
}

namespace Limits {
  export const perMinute = 60;
  export function allowed(count: number): boolean {
    return count <= perMinute;
  }
}
