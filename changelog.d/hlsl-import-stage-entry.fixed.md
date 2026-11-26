HLSL input keeps a pixel shader entry whose name does not start with `ps`. The importer marked such an entry, for
example `float4 frag(float4 c : COLOR0) : SV_Target`, as a plain function, so CrossGL synthesized an empty entry
and the translated shader wrote nothing (GLSL `void main() {}`). Every function the importer writes into a stage
block now carries `@ stage_entry`.
