# Browser application

Use the [repository quick start](../README.md) from the root. `src/` implements rendering, deformation, tracking, FBX retargeting and GSW2 transport; `backend/` provides the loopback API and opt-in native export adapters.

- [Development and tests](../docs/development.md)
- [Generation setup](../docs/generation.md)
- [Asset / pose / deformation / wire contracts](CONTRACTS.md)
- [Mixamo motion details](MIXAMO.md)
- [Configuration](../docs/configuration.md)
- [Public preview security](../docs/deployment.md)

Runtime data now lives under the ignored root `local/data`, not `web/data`. The synthetic demo has no source photos or native reference poses. Native models and their environments are never installed by the viewer quick start.
