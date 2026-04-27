# Export Notes

Export a trained checkpoint to ONNX with:

```bash
./.venv/bin/python -m model.export.export_onnx \
  --checkpoint model/artifacts/checkpoints/aptos_efficientnet_b0_best.pt \
  --output model/artifacts/model.onnx
```
