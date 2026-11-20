"""Print the browser-vs-PyTorch deviation table (steps 1 and N) from a browser_bench JSON.
Usage: python tests/summarize_bench.py logs/browser_bench_....json [variant]"""
import json, sys
r = json.load(open(sys.argv[1])); var = sys.argv[2] if len(sys.argv) > 2 else "fp32"
U = {"T_2m": ("T 2 m", "K", 1), "U_31": ("U lowest level", "m/s", 1), "V_31": ("V lowest level", "m/s", 1),
     "precip_liq_surf_mass_flux": ("liquid precip", "mm/h", 3.6e6), "ps": ("surface pressure", "Pa", 1)}
print(f"| lead | field | RMSE | max abs | bias | rel L2 |\n|---|---|---|---|---|---|")
for res in (r["results"][0], r["results"][-1]):
    for k, (lab, unit, sc) in U.items():
        s = res["cmp"][k][var]["interior"]
        print(f"| +{res['lead_min']} min | {lab} ({unit}) | {s['rmse']*sc:.3g} | {s['max_abs']*sc:.3g} | {s['bias']*sc:.2g} | {s['rel_l2']:.2g} |")
st = [x["timing"]["step_ms"] / 1000 for x in r["results"]]
print(f"\nsteps {len(st)}, mean {sum(st)/len(st):.1f} s/step (min {min(st):.1f}, max {max(st):.1f}); WASM peak {r['wasm_memory_peak_mib']:.0f} MiB; "
      f"Chrome RSS peak largest process {r['rss_peak_mib']['largest_process']:.0f} MiB, tree {r['rss_peak_mib']['chrome_tree_sum']:.0f} MiB; weights load {r['weight_load_s']:.1f} s; {r.get('chrome')}")
