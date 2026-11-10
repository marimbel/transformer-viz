# Inside a Transformer: watch one learn

An interactive explainer where a small GPT-style transformer **trains live in the browser**.
Everything (autograd, model, optimizer, visuals) is plain JavaScript. There are no dependencies and no build step.

## Sections
1. **Data pipeline**: corpus → character tokens → IDs → token + position embeddings (live heatmaps)
2. **Architecture**: clickable decoder-only transformer diagram with shapes, parameter tables and live weight heatmaps; forward/backward pass animation
3. **Self-attention**: query→key connection graph, attention matrix, every head at once, next-token predictions
4. **Neurons**: MLP drawn as a node-link network (weights or weight × activation), per token
5. **Training loop**: step-through mode (sample → forward → loss → backward → update), loss curve, gradient norms, weight/gradient/update heatmaps, generated samples
6. **Scale**: parameter calculator with GPT-2 / GPT-3 presets

## Run locally
Open `index.html` in a browser, or serve the folder:
```bash
python3 -m http.server 8000   # then visit http://localhost:8000
```

## Deploy to GitHub Pages
```bash
git init && git add . && git commit -m "Transformer training visualizer"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
Then in the repo: **Settings → Pages → Source: Deploy from a branch → main / (root)**.

## Files
- `js/engine.js`: matrix autograd (matmul, LayerNorm, causal softmax, cross-entropy…), the GPT model, Adam with gradient clipping
- `js/app.js`: all visuals and interaction
- `css/style.css`: styling

## Model details
Pre-LayerNorm decoder: token + learned position embeddings → N × [LayerNorm → causal multi-head attention → residual → LayerNorm → ReLU MLP → residual] → LayerNorm → linear head. It trains on next-character prediction with cross-entropy and Adam (β = 0.9/0.99, global grad-norm clip 1.0). The default size is d_model 32, 2 layers, 2 heads, d_ff 64, context 16, about 19K parameters. All of these can be changed on the page.
