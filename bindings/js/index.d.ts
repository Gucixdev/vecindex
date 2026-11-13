/** An id: a number up to 2^53, or a BigInt for the whole i64 range. Ids come back as numbers when they fit. */
export type Id = number | bigint;
/** A namespace: a name (hashed with key()), or a BigInt key. */
export type Ns = string | bigint;
export type Vector = ArrayLike<number> | Float32Array;
export type Hit = [Id, number];

export interface SearchOptions {
  /** Words too: BM25 over the rows' texts fused with the vector (the vector may be null: words alone). */
  text?: string | Uint8Array;
  /** One namespace, or any of several. */
  ns?: Ns | Ns[];
  /** The rows whose number is in [lo, hi]. */
  num?: [Id, Id];
  /** Labels: every one of all, at least one of any, none of none ("a,b" or an array). */
  labels?: { all?: string | string[]; any?: string | string[]; none?: string | string[] };
  /** One hit a group, the group of an id being id >> group (ids doc << 20 | chunk: 20). */
  group?: number;
}

export interface OpenOptions {
  /** Dimensions of a new store (0: it must exist). */
  dim?: number;
  metric?: 'cos' | 'cosine' | 'dot' | 'l2';
  /** Index only on build() instead of as rows come. */
  manual?: boolean;
  /** Node: the directory the store's files are in (default: the working directory). */
  dir?: string;
  /** The files: a MemoryFiles (browsers, or to export the store as bytes) or a NodeFiles. */
  files?: MemoryFiles | NodeFiles;
  /** The module's bytes or URL (default: vecindex.wasm beside index.mjs). */
  wasm?: BufferSource | URL | string;
  /** Environment the C library sees (VECINDEX_VERIFY, ...). */
  env?: Record<string, string>;
}

export class DB {
  readonly dim: number;
  readonly length: number;
  add(id: Id, vec: Vector, text?: string | Uint8Array | null): void;
  addMany(ids: Id[], vecs: Vector[] | Float32Array): void;
  update(id: Id, vec: Vector): void;
  get(id: Id): Float32Array | null;
  text(id: Id): string | null;
  text(id: Id, raw: true): Uint8Array | null;
  has(id: Id): boolean;
  delete(id: Id): boolean;
  deleteRange(lo: Id, hi: Id): number;
  setTag(id: Id, tag: Id): void;
  tag(id: Id): bigint | null;
  setNum(id: Id, num: Id): void;
  num(id: Id): Id | null;
  setNs(id: Id, ns: Ns): void;
  ns(id: Id): bigint | null;
  countNs(ns: Ns): number;
  deleteNs(ns: Ns): number;
  label(name: string, parent?: string): void;
  addLabels(id: Id, labels: string | string[]): void;
  removeLabels(id: Id, labels: string | string[]): void;
  search(vec: Vector | null, k?: number, opts?: SearchOptions): Hit[];
  /** The k rows with the highest exact score (the same as scoring every row); checked: rows scored exactly. */
  searchExact(vec: Vector, k?: number): { hits: Hit[]; checked: number };
  build(): void;
  save(path?: string): void;
  sync(): void;
  close(): void;
}

export class Library {
  static load(opts?: { files?: MemoryFiles | NodeFiles; wasm?: BufferSource | URL | string; env?: Record<string, string> }): Promise<Library>;
  version(): string;
  key(name: string): bigint;
  open(path: string | null, opts?: { dim?: number; metric?: 'cos' | 'cosine' | 'dot' | 'l2'; manual?: boolean }): DB;
}

export class MemoryFiles {
  constructor();
  exportFile(path: string): Uint8Array | null;
  importFile(path: string, bytes: ArrayBufferLike | ArrayLike<number>): void;
}

export class NodeFiles {
  constructor(fs: unknown, root: string);
}

export function open(path: string | null, opts?: OpenOptions): Promise<DB>;
export function version(): Promise<string>;
