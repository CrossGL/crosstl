// Check WGSL files with Tint, the WGSL compiler of Dawn, through Dawn's Node.js bindings (npm webgpu).
//
// usage: node tint.mjs WEBGPU_PACKAGE FILE.wgsl...
//
// WEBGPU_PACKAGE is the directory of the installed webgpu package. The device runs on Dawn's null backend, so no GPU
// is needed, and enables every optional WGSL feature the adapter offers, so that an `enable` directive is checked
// against Tint's rules for its extension. Each error is printed as FILE:LINE:COLUMN: error: MESSAGE; the exit status
// is 1 when a file has an error and 2 when no device can be created.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

// The device features behind WGSL's enable extensions (WGSL section 4.1.1) and the dual-source blending one.
const FEATURES = ['shader-f16', 'subgroups', 'clip-distances', 'dual-source-blending', 'primitive-index'];

const [packageDirectory, ...files] = process.argv.slice(2);
const { create, globals } = await import(pathToFileURL(join(packageDirectory, 'index.js')).href);
Object.assign(globalThis, globals);
const gpu = create(['backend=null', 'enable-dawn-features=allow_unsafe_apis']);
const adapter = await gpu.requestAdapter();
if (!adapter) {
  console.error('error: Dawn offers no null-backend adapter');
  process.exit(2);
}
const device = await adapter.requestDevice({
  requiredFeatures: FEATURES.filter((feature) => adapter.features.has(feature)),
});
let rejected = false;
for (const file of files) {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code: readFileSync(file, 'utf8') });
  const { messages } = await module.getCompilationInfo();
  const scoped = await device.popErrorScope();
  const errors = messages.filter((message) => message.type === 'error');
  for (const error of errors) {
    console.error(`${file}:${error.lineNum}:${error.linePos}: error: ${error.message}`);
  }
  if (scoped && !errors.length) {
    console.error(`${file}: error: ${scoped.message}`);
  }
  rejected ||= Boolean(errors.length || scoped);
}
device.destroy();
process.exit(rejected ? 1 : 0);
