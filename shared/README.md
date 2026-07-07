# Shared Runtime Assets

This directory is the versioned anchor for runtime assets shared by scenario
subprojects.

- `hitran_cache/` is the shared HITRAN spectrum cache for `hydrogen_ng` and
  `syngas`; `tunnel_ventilation` may also use it when the HITRAN backend is
  selected.
- Cache contents are generated locally and are ignored by Git.
- `_archived/` is reserved for local archived runtime artifacts and is ignored
  by Git.

Typical CLI override from a scenario subproject:

```powershell
--hitran-cache-root ../shared/hitran_cache
```
