// Shared (browser worker + Node) loader for the STRATAQ8 weight file.
// The whole file is placed in WASM memory once; tensors are bound by name
// to the engine with pointers into that buffer (all tensor offsets are 64-aligned).

export function parseHeader(u8) {
  const magic = new TextDecoder().decode(u8.slice(0, 8));
  if (magic !== 'STRATAQ8') throw new Error('not a STRATAQ8 file');
  const hlen = new DataView(u8.buffer, u8.byteOffset + 8, 4).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(u8.slice(12, 12 + hlen)));
  header.dataStart = Math.ceil((12 + hlen) / 64) * 64;
  return header;
}

// Copies `chunks` (an async iterable of Uint8Array, e.g. a fetch body reader)
// into a single WASM allocation of `total` bytes. Returns the base pointer.
export async function streamIntoWasm(M, chunks, total, onProgress) {
  const base = M._st_malloc(total);
  if (!base) throw new Error(`WASM allocation of ${(total / 2 ** 20).toFixed(0)} MiB failed`);
  let off = 0;
  for await (const c of chunks) {
    M.HEAPU8.set(c, base + off);  // re-read HEAPU8: memory may have grown
    off += c.byteLength;
    if (onProgress) onProgress(off, total);
  }
  if (off !== total) throw new Error(`weight stream ended at ${off} of ${total} bytes`);
  return base;
}

export function bindWeights(M, base, header) {
  M._st_set_group(header.meta.group_size || 0);
  let bound = 0;
  const unbound = [];
  for (const [name, t] of Object.entries(header.tensors)) {
    if (name.startsWith('input_norm') || name.startsWith('target_norm')) continue;
    const p = M.stringToNewUTF8(name);
    const ok = M._st_bind(p, base + header.dataStart + t.offset);
    M._free(p);
    if (ok) bound++; else unbound.push(name);
  }
  const missing = M._st_check_bound();
  if (missing) throw new Error(`${missing} engine weights not bound; unknown names: ${unbound.slice(0, 5)}`);
  return bound;
}

export function f32View(M, base, header, name) {
  const t = header.tensors[name];
  const n = t.shape.reduce((a, b) => a * b, 1);
  return new Float32Array(M.HEAPU8.buffer, base + header.dataStart + t.offset, n).slice();
}
