# Real background photos

Drop photos here and list them in `manifest.json` — the degradation stack will blend
generated pages onto them instead of procedural textures (much closer to reality).

- **desk**: photos of your table/desk surface (used behind the page in the `photo` preset's scene stage)
- **paper**: photos of *blank* notebook pages, straight-on, evenly lit (multiply-blended onto the page as texture)

```json
{
  "desk": ["desk1.jpg", "desk2.jpg"],
  "paper": ["page1.jpg", "page2.jpg", "page3.jpg"]
}
```

Tips: 5–10 of each is plenty; shoot the *actual notebook* the model will read; avoid
text/objects in frame; any resolution (they're cover-scaled with random crop/flip).
No manifest.json → procedural fallback, nothing breaks.
