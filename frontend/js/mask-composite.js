/**
 * Mask Compositor Module
 *
 * Reads mask-group rows from LayerControlPanel via `panel.getMaskRows()` and
 * renders a single yellow binary-mask overlay on the map. Standalone (non-
 * grouped) layer rows do NOT participate — masks are composed only by
 * dragging two binary rows together to form a group.
 *
 * Per group composition (left-to-right, in the order items appear in the row):
 *   - items[0]'s alpha mask seeds the running composite.
 *   - operators[i] joins the running composite with items[i+1]'s mask:
 *       inc (∩) : composite = composite AND mask
 *       exc (−) : composite = composite AND NOT mask
 *       add (∪) : composite = composite OR mask
 * A pixel is "in" if its source PNG alpha > MASK_ALPHA_THRESHOLD.
 *
 * If the panel holds multiple groups, each is composed independently and the
 * results are combined by simple union (OR). Group-level visibility is
 * derived from the first item's `visible` flag (group toggle propagates to
 * all items in the panel).
 *
 * Cloud + local: works identically. Source URLs come from
 * `entry.leafletLayer._url` (Leaflet imageOverlay) or `.getElement().src`
 * fallback. GeoRaster-backed layers expose neither and are skipped.
 */

const MASK_ALPHA_THRESHOLD = 32;
const COMPOSITE_RENDER_W = 768;
const COMPOSITE_FILL = [255, 220, 0, 180]; // RGBA — yellow, semi-transparent

class MaskCompositor {
    constructor(mapManager, panel) {
        this.mapManager = mapManager;
        this.panel = panel;
        this._compositeOverlay = null;
        this._sourceCache = new Map(); // url → HTMLImageElement
        this._pending = false;
        this._wire();
    }

    _wire() {
        const trigger = () => this.scheduleRecompute();
        window.addEventListener('mask:registry-changed', trigger);
        window.addEventListener('layers:cleared', trigger);
    }

    scheduleRecompute() {
        if (this._pending) return;
        this._pending = true;
        Promise.resolve().then(() => {
            this._pending = false;
            this.recompute().catch(err =>
                console.error('[MaskCompositor] recompute error:', err));
        });
    }

    _activeGroups() {
        if (!this.panel || typeof this.panel.getMaskRows !== 'function') return [];
        const rows = this.panel.getMaskRows();
        const groups = [];
        for (const row of rows) {
            if (!row || row.kind !== 'group') continue;
            if (row.visible === false) continue;
            if (!Array.isArray(row.items) || row.items.length < 2) continue;
            const items = row.items.filter(e => e && e.leafletLayer);
            if (items.length < 2) continue;
            // Operators correspond to original positions; if some items were
            // filtered out we'd lose alignment, so require all-present.
            if (items.length !== row.items.length) continue;
            groups.push(row);
        }
        return groups;
    }

    _entrySource(entry) {
        const layer = entry.leafletLayer;
        if (!layer) return null;
        const bounds = (typeof layer.getBounds === 'function') ? layer.getBounds() : null;
        if (layer._url) return { url: layer._url, bounds };
        const el = (typeof layer.getElement === 'function') ? layer.getElement() : null;
        if (el && el.tagName === 'IMG' && el.src) {
            return { url: el.src, bounds };
        }
        // GeoRasterLayer / canvas-based layers — skipped.
        return null;
    }

    async _loadImage(url) {
        if (this._sourceCache.has(url)) return this._sourceCache.get(url);
        const img = new Image();
        img.crossOrigin = 'anonymous';
        const p = new Promise((resolve, reject) => {
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('img load failed: ' + url));
        });
        img.src = url;
        await p;
        this._sourceCache.set(url, img);
        return img;
    }

    /**
     * Compose one group into a Uint8Array mask of length W*H.
     * Returns { mask, bounds, W, H } or null on failure.
     */
    async _composeGroup(group) {
        const sources = await Promise.all(group.items.map(async (entry) => {
            const s = this._entrySource(entry);
            if (!s || !s.bounds) return null;
            try {
                const img = await this._loadImage(s.url);
                return { entry, img, bounds: s.bounds };
            } catch (err) {
                console.warn('[MaskCompositor] skipping item, image failed:', entry.id, err);
                return null;
            }
        }));
        const valid = sources.filter(Boolean);
        if (valid.length < 2) return null;

        const first = valid[0].img;
        const groupBounds = valid[0].bounds;
        const aspect = (first.naturalHeight || first.height) / (first.naturalWidth || first.width);
        const W = Math.min(COMPOSITE_RENDER_W, first.naturalWidth || first.width || COMPOSITE_RENDER_W);
        const H = Math.max(1, Math.round(W * aspect));

        const scratch = document.createElement('canvas');
        scratch.width = W; scratch.height = H;
        const sctx = scratch.getContext('2d', { willReadFrequently: true });

        let composite = null;
        for (let k = 0; k < valid.length; k++) {
            const { entry, img } = valid[k];
            sctx.clearRect(0, 0, W, H);
            try {
                sctx.drawImage(img, 0, 0, W, H);
            } catch (err) {
                console.warn('[MaskCompositor] drawImage failed (CORS?):', entry.id, err);
                continue;
            }
            let data;
            try {
                data = sctx.getImageData(0, 0, W, H).data;
            } catch (err) {
                console.warn('[MaskCompositor] getImageData failed (tainted):', entry.id, err);
                continue;
            }
            const mask = new Uint8Array(W * H);
            for (let i = 0, j = 3; i < mask.length; i++, j += 4) {
                mask[i] = data[j] > MASK_ALPHA_THRESHOLD ? 1 : 0;
            }
            if (composite === null) {
                composite = mask;
                continue;
            }
            const role = group.operators[k - 1] || 'inc';
            if (role === 'inc') {
                for (let i = 0; i < composite.length; i++) composite[i] = composite[i] & mask[i];
            } else if (role === 'exc') {
                for (let i = 0; i < composite.length; i++) composite[i] = composite[i] & (1 - mask[i]);
            } else if (role === 'add') {
                for (let i = 0; i < composite.length; i++) composite[i] = composite[i] | mask[i];
            }
        }
        if (composite === null) return null;
        return { mask: composite, bounds: groupBounds, W, H };
    }

    async recompute() {
        const map = this.mapManager?.map;
        if (!map) return;

        const groups = this._activeGroups();
        if (groups.length === 0) {
            this._removeOverlay();
            return;
        }

        const composed = [];
        for (const g of groups) {
            const r = await this._composeGroup(g);
            if (r) composed.push(r);
        }
        if (composed.length === 0) {
            this._removeOverlay();
            return;
        }

        // Use the first group's geometry as the canvas extent. Subsequent
        // groups' masks are unioned in by element-wise OR. If their pixel
        // shapes differ they still align by index (each group composed at
        // its own first-item resolution); we resample by re-drawing into a
        // shared-size canvas only if shapes mismatch.
        const head = composed[0];
        const W = head.W, H = head.H;
        const finalMask = new Uint8Array(head.mask);

        for (let g = 1; g < composed.length; g++) {
            const cur = composed[g];
            if (cur.W === W && cur.H === H) {
                for (let i = 0; i < finalMask.length; i++) finalMask[i] |= cur.mask[i];
            } else {
                // Shape mismatch: rasterize cur.mask onto a head-sized canvas.
                const tmpIn = document.createElement('canvas');
                tmpIn.width = cur.W; tmpIn.height = cur.H;
                const tic = tmpIn.getContext('2d');
                const id = tic.createImageData(cur.W, cur.H);
                for (let i = 0, j = 0; i < cur.mask.length; i++, j += 4) {
                    if (cur.mask[i]) { id.data[j + 3] = 255; }
                }
                tic.putImageData(id, 0, 0);
                const tmpOut = document.createElement('canvas');
                tmpOut.width = W; tmpOut.height = H;
                const toc = tmpOut.getContext('2d', { willReadFrequently: true });
                toc.drawImage(tmpIn, 0, 0, W, H);
                const out = toc.getImageData(0, 0, W, H).data;
                for (let i = 0, j = 3; i < finalMask.length; i++, j += 4) {
                    if (out[j] > MASK_ALPHA_THRESHOLD) finalMask[i] = 1;
                }
            }
        }

        const out = document.createElement('canvas');
        out.width = W; out.height = H;
        const octx = out.getContext('2d');
        const outImage = octx.createImageData(W, H);
        const [R, G, B, A] = COMPOSITE_FILL;
        for (let i = 0, j = 0; i < finalMask.length; i++, j += 4) {
            if (finalMask[i]) {
                outImage.data[j] = R;
                outImage.data[j + 1] = G;
                outImage.data[j + 2] = B;
                outImage.data[j + 3] = A;
            }
        }
        octx.putImageData(outImage, 0, 0);
        const dataUrl = out.toDataURL('image/png');

        this._removeOverlay();
        this._compositeOverlay = L.imageOverlay(dataUrl, head.bounds, {
            opacity: 1.0,
            interactive: false,
            pane: 'analysisPane',
        });
        this._compositeOverlay.addTo(map);
        const el = (typeof this._compositeOverlay.getElement === 'function')
            ? this._compositeOverlay.getElement()
            : this._compositeOverlay._image;
        if (el) {
            el.style.zIndex = '5000';
            el.classList.add('mask-composite-overlay');
        }
    }

    _removeOverlay() {
        const map = this.mapManager?.map;
        if (this._compositeOverlay && map) {
            try { map.removeLayer(this._compositeOverlay); } catch (_) { /* no-op */ }
        }
        this._compositeOverlay = null;
    }
}

window.MaskCompositor = MaskCompositor;
