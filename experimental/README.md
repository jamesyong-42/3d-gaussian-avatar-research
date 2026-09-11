# Earlier experiments and Unity scaffolding

These are retained design experiments, **not** the browser runtime, a production SDK or a ready-to-import Quest application.

`python/` and `csharp/` implement an earlier protocol with **50 expression channels** and different packet/layout identifiers. The maintained web pipeline uses **GSW2, 55 joints and 100 expression channels**. Do not connect these implementations directly without a deliberate port and cross-language fixtures.

`unity/` contains early binder/entity/recording sketches and a Gaussian LBS compute shader. It lacks a validated complete renderer, imported licensed rig assets, Meta device integration, stereo/Quest build verification and production lifecycle handling. Desktop browser measurements are not Unity or standalone-headset results.

Run the earlier Python-only tests with `python experimental/python/run_tests.py` (NumPy required; use the API `.venv` interpreter). The C# solution has its own project dependencies. These experiments are outside the default public CI promise; update the authoritative [web contracts](../web/CONTRACTS.md) before promoting any component.

Next steps: reproduce the current asset/pose math in C#, explicitly convert handedness/units, validate all native fixtures, integrate stereo splat rendering, then calibrate actual Meta head/hand/face signals on target hardware. See [the roadmap](../docs/research.md).
