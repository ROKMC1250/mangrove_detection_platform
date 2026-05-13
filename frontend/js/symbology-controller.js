/**
 * Symbology Controller — Visualization (server-side, GEE) + Rendering (client-side CSS+SVG filter).
 *
 * One instance per prefix ('s2', 's1', 'local'). Each instance owns the markup at
 * `.symbology-block[data-prefix="<prefix>"]`. The controller is intentionally thin:
 *
 *   - Visualization state (mode/min/max/pct_low/pct_high/bands) is mirrored into the
 *     existing hidden inputs (`#${prefix}-min`, `#${prefix}-max`, `#${prefix}-pct-*`)
 *     so legacy code paths (e.g. `getS2VisualizationParams()`, `applyVisualizationToAllImages`,
 *     and `local-image-controller.js`) keep working unchanged.
 *   - Rendering state is applied directly on the tile layer's container DOM via
 *     `mapManager.applyRenderingFilter()`. Updates are immediate (60fps drag).
 *
 * Layers belonging to this prefix are recognized by id pattern (`preview-${imageId}`).
 * S1 vs S2 disambiguation matches the rule used in `applyVisualizationToAllImages`.
 */
(function () {
    'use strict';

    const DEFAULT_RENDER = { gamma: 1.0, contrast: 1.0, brightness: 1.0, saturation: 1.0 };

    class SymbologyController {
        constructor(prefix) {
            this.prefix = prefix; // 's2' | 's1' | 'local'
            this.block = document.querySelector(`.symbology-block[data-prefix="${prefix}"]`);
            if (!this.block) {
                console.warn(`[Symbology:${prefix}] no .symbology-block found`);
                return;
            }

            const ds = this.block.dataset;
            this.defaults = {
                min: parseFloat(ds.defaultMin),
                max: parseFloat(ds.defaultMax),
                pctLow: parseFloat(ds.defaultPctLow),
                pctHigh: parseFloat(ds.defaultPctHigh),
            };

            // Per-band Min/Max state. Cloud paths (s2/s1) send these as 3-element arrays
            // to ee.Image.visualize() so each RGB band gets its own range. Local path
            // currently has a single-stretch backend (gpu-stretch), so the controller
            // *links* the three channels for local — changing any input syncs the others.
            this.state = {
                mode: 'minmax',                              // 'minmax' | 'percentile'
                bands: {
                    r: { min: this.defaults.min, max: this.defaults.max },
                    g: { min: this.defaults.min, max: this.defaults.max },
                    b: { min: this.defaults.min, max: this.defaults.max },
                },
                pctLow: this.defaults.pctLow,
                pctHigh: this.defaults.pctHigh,
                rendering: { ...DEFAULT_RENDER },
            };
            // All prefixes now support true per-band stretch:
            //   - cloud (s2/s1) via `ee.Image.visualize(min=[..], max=[..])`
            //   - local via `/api/local/uploaded/gpu-stretch-multi`
            this.linkChannels = false;

            this._wireSections();
            this._wireModeToggle();
            this._wireMinMaxInputs();
            this._wirePercentileSliders();
            this._wireRenderingControls();
            this._wireSectionActions();
            this._wireLayerEvents();

            this._writeHiddenInputs();
            this._applyRenderingToAllMatchingLayers();
        }

        // ---------- Wire helpers ----------

        _qs(sel)  { return this.block.querySelector(sel); }
        _qsa(sel) { return Array.from(this.block.querySelectorAll(sel)); }

        _wireSections() {
            this._qsa('.symbology-section-header .symbology-toggle-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const sec = btn.closest('.symbology-section');
                    if (!sec) return;
                    sec.classList.toggle('collapsed');
                    const chevron = btn.querySelector('.symbology-chevron');
                    if (chevron) chevron.textContent = sec.classList.contains('collapsed') ? '▸' : '▾';
                });
            });
        }

        _wireModeToggle() {
            this._qsa('.mode-toggle-option').forEach(opt => {
                opt.addEventListener('click', () => {
                    const mode = opt.dataset.mode;
                    if (!mode || mode === this.state.mode) return;
                    this.state.mode = mode;
                    this._qsa('.mode-toggle-option').forEach(o => o.classList.toggle('active', o === opt));
                    this._qsa('.mode-panel').forEach(p => p.classList.toggle('hidden', p.dataset.modePanel !== mode));
                    this._emitVisualizationChanged('mode');
                });
            });
        }

        _wireMinMaxInputs() {
            const inputs = this._qsa('.symbology-num-input[data-channel]');
            inputs.forEach(input => {
                const ch = input.dataset.channel;     // 'r' | 'g' | 'b'
                const field = input.dataset.field;    // 'min' | 'max'
                const onChange = () => {
                    const v = parseFloat(input.value);
                    if (!Number.isFinite(v)) return;
                    if (this.linkChannels) {
                        // Local: keep R/G/B in sync (single-stretch backend).
                        ['r','g','b'].forEach(c => { this.state.bands[c][field] = v; });
                        // Reflect in DOM siblings.
                        this._qsa(`.symbology-num-input[data-field="${field}"]`).forEach(el => {
                            if (el !== input) el.value = String(v);
                        });
                    } else {
                        this.state.bands[ch][field] = v;
                    }
                    this._writeHiddenInputs();
                    this._emitVisualizationChanged('input');
                };
                // Use 'input' so live drag of arrow buttons triggers updates;
                // outer code (e.g. local instant-apply) debounces if needed.
                input.addEventListener('change', onChange);
                input.addEventListener('input', onChange);
            });
        }

        _wirePercentileSliders() {
            // The slider DOM is the existing #${prefix}-pct-{low,high} pair (no longer wrapped
            // in `.percentile-stretch`, but with the same ids). Local controller has its own
            // logic using band percentiles; for s2/s1 we map slider% to the data range linearly.
            const lowSlider = document.getElementById(`${this.prefix}-pct-low`);
            const highSlider = document.getElementById(`${this.prefix}-pct-high`);
            const lowVal = document.getElementById(`${this.prefix}-pct-low-val`);
            const highVal = document.getElementById(`${this.prefix}-pct-high-val`);
            if (!lowSlider || !highSlider) return;

            // NOTE: We deliberately do NOT call `_writeHiddenInputs` from percentile-slider
            // handlers. The legacy `local-image-controller` has its own listener on these
            // sliders that maps slider% → real band percentiles → hidden min/max. Writing
            // state.min/max from here would clobber that. For cloud (s2/s1), the hidden
            // min/max are also irrelevant in percentile mode because the server recomputes
            // them when `stretch_mode='percentile'` is sent.
            const onSlider = () => {
                let lo = parseFloat(lowSlider.value);
                let hi = parseFloat(highSlider.value);
                if (lo > hi) { lowSlider.value = hi; lo = hi; }
                if (hi < lo) { highSlider.value = lo; hi = lo; }
                if (lowVal)  lowVal.value  = lo.toFixed(1);
                if (highVal) highVal.value = hi.toFixed(1);
                this.state.pctLow = lo;
                this.state.pctHigh = hi;
                this._emitVisualizationChanged('percentile');
            };
            const onInput = () => {
                let lo = parseFloat(lowVal?.value);
                let hi = parseFloat(highVal?.value);
                if (!Number.isFinite(lo)) lo = 0;
                if (!Number.isFinite(hi)) hi = 100;
                lo = Math.max(0, Math.min(100, lo));
                hi = Math.max(0, Math.min(100, hi));
                if (lo > hi) lo = hi;
                lowSlider.value = lo;
                highSlider.value = hi;
                this.state.pctLow = lo;
                this.state.pctHigh = hi;
                this._emitVisualizationChanged('percentile');
            };

            lowSlider.addEventListener('input', onSlider);
            highSlider.addEventListener('input', onSlider);
            if (lowVal)  lowVal.addEventListener('change', onInput);
            if (highVal) highVal.addEventListener('change', onInput);
        }

        _wireRenderingControls() {
            this._qsa('.render-row').forEach(row => {
                const key = row.dataset.key;
                const slider = row.querySelector('.render-slider');
                const num = row.querySelector('.render-num');

                const propagate = (v) => {
                    const clamped = Math.max(parseFloat(row.dataset.min), Math.min(parseFloat(row.dataset.max), v));
                    this.state.rendering[key] = clamped;
                    if (slider) slider.value = String(clamped);
                    if (num)    num.value    = clamped.toFixed(2);
                    this._applyRenderingToAllMatchingLayers();
                };

                if (slider) {
                    slider.addEventListener('input', () => {
                        const v = parseFloat(slider.value);
                        if (Number.isFinite(v)) propagate(v);
                    });
                }
                if (num) {
                    num.addEventListener('change', () => {
                        const v = parseFloat(num.value);
                        if (Number.isFinite(v)) propagate(v);
                    });
                }
                // Double-click on slider resets that single control to default.
                if (slider) {
                    slider.addEventListener('dblclick', () => {
                        const def = parseFloat(row.dataset.default);
                        if (Number.isFinite(def)) propagate(def);
                    });
                }
            });
        }

        _wireSectionActions() {
            this._qsa('.symbology-icon-btn').forEach(btn => {
                const action = btn.dataset.action;
                if (!action) return;
                btn.addEventListener('click', () => this._handleAction(action));
            });
        }

        _wireLayerEvents() {
            // Re-apply rendering whenever a layer this controller cares about appears.
            //  - cloud (s2/s1): tile layers (`type: 'tile'`)
            //  - local: image overlays (`type: 'image'`) — local-image-controller dispatches
            //           layer:added for its base RGB composite + band overlays.
            window.addEventListener('layer:added', (e) => {
                const detail = e.detail || {};
                const isCloudType = detail.type === 'tile' && (this.prefix === 's2' || this.prefix === 's1');
                const isLocalType = detail.type === 'image' && this.prefix === 'local';
                if (!isCloudType && !isLocalType) return;
                if (!this._layerMatchesPrefix(detail.id)) return;
                if (this.prefix === 'local') {
                    // Defer one tick: imageOverlay's <img> isn't always in the DOM at the
                    // exact moment the event fires (Leaflet adds the element synchronously
                    // but layout/styles bind after). RAF avoids dropped first-paint filter.
                    requestAnimationFrame(() => this._applyRenderingToLocalOverlays());
                } else if (detail.layer) {
                    window.mapManager.applyRenderingFilter(detail.layer, this.state.rendering, this.prefix);
                }
            });
        }

        // ---------- Actions ----------

        _handleAction(action) {
            switch (action) {
                case 'reset-vis':   return this._resetVisualization();
                case 'reset-render':return this._resetRendering();
                case 'reset-bands': return this._resetBands();
                case 'auto':        return this._autoStretch();
            }
        }

        _resetBands() {
            // Restore RGB slots to defaults declared on the section (data-default-bands).
            // Local prefix has no fixed default band names — pick the first three available
            // chips in the pool. Visualization & rendering states are intentionally NOT
            // touched: this reset is scoped to the bands section only.
            const section = this._qs('.symbology-section[data-section="bands"]');
            const declared = section?.dataset.defaultBands;
            let bands;
            if (declared) {
                bands = declared.split(',').map(s => s.trim()).filter(Boolean);
            } else {
                const pool = document.getElementById(`${this.prefix}-band-pool`);
                const chips = pool ? Array.from(pool.querySelectorAll('.band-chip')).slice(0, 3) : [];
                bands = chips.map(c => c.dataset.band).filter(Boolean);
            }
            if (bands.length < 3) return;
            ['r','g','b'].forEach((ch, i) => {
                const slot = document.getElementById(`${this.prefix}-slot-${ch}`);
                if (!slot) return;
                const lbl = slot.querySelector('.slot-band');
                const band = bands[i];
                slot.dataset.band = band;
                if (lbl) {
                    lbl.dataset.band = band;
                    // Mirror the existing display-shortening rule used by other handlers.
                    lbl.textContent = band.replace(/^Band\s*/i, '').replace('after_', '').replace('.tif', '');
                }
            });
            // For local mode, recompose immediately so the change is visible.
            if (this.prefix === 'local') {
                this._emitVisualizationChanged('bands-reset');
            }
        }

        _resetVisualization() {
            this.state.mode = 'minmax';
            ['r','g','b'].forEach(c => {
                this.state.bands[c].min = this.defaults.min;
                this.state.bands[c].max = this.defaults.max;
            });
            this.state.pctLow = this.defaults.pctLow;
            this.state.pctHigh = this.defaults.pctHigh;

            this._qsa('.symbology-num-input[data-channel]').forEach(el => {
                const field = el.dataset.field;
                el.value = String(field === 'min' ? this.defaults.min : this.defaults.max);
            });

            const lowSlider = document.getElementById(`${this.prefix}-pct-low`);
            const highSlider = document.getElementById(`${this.prefix}-pct-high`);
            const lowVal = document.getElementById(`${this.prefix}-pct-low-val`);
            const highVal = document.getElementById(`${this.prefix}-pct-high-val`);
            if (lowSlider) lowSlider.value = String(this.defaults.pctLow);
            if (highSlider) highSlider.value = String(this.defaults.pctHigh);
            if (lowVal) lowVal.value = this.defaults.pctLow.toFixed(1);
            if (highVal) highVal.value = this.defaults.pctHigh.toFixed(1);

            this._qsa('.mode-toggle-option').forEach(o => o.classList.toggle('active', o.dataset.mode === 'minmax'));
            this._qsa('.mode-panel').forEach(p => p.classList.toggle('hidden', p.dataset.modePanel !== 'minmax'));

            this._writeHiddenInputs();
            this._emitVisualizationChanged('reset');
        }

        _resetRendering() {
            this.state.rendering = { ...DEFAULT_RENDER };
            this._qsa('.render-row').forEach(row => {
                const key = row.dataset.key;
                const def = parseFloat(row.dataset.default);
                const slider = row.querySelector('.render-slider');
                const num = row.querySelector('.render-num');
                if (slider) slider.value = String(def);
                if (num) num.value = def.toFixed(2);
                this.state.rendering[key] = def;
            });
            this._applyRenderingToAllMatchingLayers();
        }

        async _autoStretch() {
            try {
                if (this.prefix === 'local') {
                    return this._autoStretchLocal();
                }
                return this._autoStretchCloud();
            } catch (err) {
                console.error('[Symbology] auto-stretch failed:', err);
            }
        }

        async _autoStretchCloud() {
            const aoi = this._getAOI();
            if (!aoi || !aoi.bbox) {
                this._notify('Define an AOI first', 'warning');
                return;
            }
            const layers = this._collectMatchingLayers();
            if (layers.length === 0) {
                this._notify('Display an image first to auto-stretch', 'warning');
                return;
            }
            const layerId = layers[0].id;
            const itemId = layerId.replace(/^preview-/, '');
            const bands = this._currentBands();

            const pctLow = this.state.pctLow;
            const pctHigh = this.state.pctHigh;

            const resp = await fetch('/api/compute-stretch-stats', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    item_id: itemId,
                    bbox: aoi.bbox,
                    geometry: aoi.geometry,
                    bands,
                    sensor: this.prefix === 's1' ? 's1' : 's2',
                    pct_low: pctLow,
                    pct_high: pctHigh,
                }),
            });
            if (!resp.ok) {
                const detail = await resp.text().catch(() => '');
                console.error('compute-stretch-stats failed', resp.status, detail);
                this._notify('Auto-stretch request failed', 'error');
                return;
            }
            const data = await resp.json();
            const lows = [], highs = [];
            Object.values(data.bands || {}).forEach(b => {
                if (b && b.p_low != null) lows.push(b.p_low);
                if (b && b.p_high != null) highs.push(b.p_high);
            });
            if (!lows.length || !highs.length) {
                this._notify('No stretch stats returned', 'warning');
                return;
            }
            const lo = Math.min(...lows);
            const hi = Math.max(...highs);
            this._setMinMax(lo, hi);
            this._notify(`Auto: ${lo.toFixed(1)} – ${hi.toFixed(1)}`, 'info', 2000);
        }

        _autoStretchLocal() {
            // Local controller exposes per-band percentiles via window.localImageController.percentiles.
            // The percentile slider in cumulative mode already maps to data-domain min/max for local;
            // here we just re-trigger the mapping with current pct values.
            const lc = window.localImageController;
            if (!lc || !lc.percentiles) {
                this._notify('Upload and load a raster first', 'warning');
                return;
            }
            const lo = this.state.pctLow;
            const hi = this.state.pctHigh;
            const loKey = String(Math.max(1, Math.round(lo)));
            const hiKey = String(Math.min(99, Math.round(hi)));
            const minVal = lc.percentiles[loKey];
            const maxVal = lc.percentiles[hiKey];
            if (minVal == null || maxVal == null) {
                this._notify('Percentile data missing', 'warning');
                return;
            }
            this._setMinMax(Math.round(minVal), Math.round(maxVal));
            this._notify(`Auto: ${minVal.toFixed(1)} – ${maxVal.toFixed(1)}`, 'info', 2000);
        }

        _setMinMax(lo, hi) {
            // Apply to all 3 channels uniformly (Auto button result is a single range).
            ['r','g','b'].forEach(c => {
                this.state.bands[c].min = lo;
                this.state.bands[c].max = hi;
            });
            this._qsa('.symbology-num-input[data-channel][data-field="min"]').forEach(el => el.value = String(lo));
            this._qsa('.symbology-num-input[data-channel][data-field="max"]').forEach(el => el.value = String(hi));
            this._writeHiddenInputs();
            this._emitVisualizationChanged('auto');
        }

        // ---------- Bridge to legacy hidden inputs ----------

        _writeHiddenInputs() {
            // Hidden inputs are scalars (legacy callers like getS2VisualizationParams expect
            // scalar). Use R-channel value as the canonical scalar — for cloud, when per-band
            // arrays are sent via getRequestPatch, the backend uses those instead.
            const minH = document.getElementById(`${this.prefix}-min`);
            const maxH = document.getElementById(`${this.prefix}-max`);
            if (minH) minH.value = String(this.state.bands.r.min);
            if (maxH) maxH.value = String(this.state.bands.r.max);
        }

        _emitVisualizationChanged(source) {
            // Custom event for downstream listeners (e.g. local-image-controller's
            // instant-apply hook). Cloud paths rely on Apply button explicitly.
            try {
                window.dispatchEvent(new CustomEvent('symbology:visualization-changed', {
                    detail: { prefix: this.prefix, source, state: this.state }
                }));
            } catch (e) {}
        }

        // ---------- Misc helpers ----------

        _layerMatchesPrefix(layerId) {
            if (!layerId) return false;
            // Local controller emits ids like 'local-base-image' (no 'preview-' prefix);
            // cloud previews use 'preview-<satellite-id>'.
            if (this.prefix === 'local') return layerId.startsWith('local-');
            if (!layerId.startsWith('preview-')) return false;
            const imageId = layerId.replace(/^preview-/, '');
            const isS1 = imageId.startsWith('S1A_') || imageId.startsWith('S1B_') || imageId.includes('S1_GRD');
            if (this.prefix === 's1') return isS1;
            if (this.prefix === 's2') return !isS1 && !imageId.startsWith('local-');
            return false;
        }

        _collectMatchingLayers() {
            const tileLayers = (window.mapManager && window.mapManager.tileLayers) || {};
            return Object.entries(tileLayers)
                .filter(([id]) => this._layerMatchesPrefix(id))
                .map(([id, layer]) => ({ id, layer }));
        }

        _applyRenderingToAllMatchingLayers() {
            // Cloud paths (s2 / s1): tile layers live in mapManager.tileLayers and use Leaflet's
            // tile container. Local path: image overlays — `geoOverlay` on the main map for
            // load-image mode, `currentOverlay` on the CRS.Simple `localMap` for band-
            // registration mode.
            if (this.prefix === 'local') {
                this._applyRenderingToLocalOverlays();
                return;
            }
            const layers = this._collectMatchingLayers();
            for (const { layer } of layers) {
                window.mapManager.applyRenderingFilter(layer, this.state.rendering, this.prefix);
            }
        }

        _buildLocalCssFilter() {
            const s = this.state.rendering;
            const gamma = Math.max(0.05, Number.isFinite(s.gamma) ? s.gamma : 1);
            const exponent = 1.0 / gamma;
            const filterEl = document.getElementById(`symbology-gamma-${this.prefix}`);
            if (filterEl) {
                filterEl.querySelectorAll('feFuncR, feFuncG, feFuncB')
                    .forEach(fn => fn.setAttribute('exponent', String(exponent)));
            }
            return `url(#symbology-gamma-${this.prefix}) brightness(${s.brightness}) contrast(${s.contrast}) saturate(${s.saturation})`;
        }

        _applyRenderingToLocalOverlays() {
            const cssFilter = this._buildLocalCssFilter();
            const lc = window.localImageController;

            // Load-image mode: overlay lives on the main map. Apply filter directly to the
            // overlay's <img> element so it doesn't bleed onto S2/S1 preview tiles that may
            // share the same overlay-pane.
            const apply = (overlay) => {
                if (!overlay || typeof overlay.getElement !== 'function') return;
                const el = overlay.getElement();
                if (el) {
                    el.style.filter = cssFilter;
                    el.style.willChange = 'filter';
                }
            };
            if (lc) {
                apply(lc.geoOverlay);
                if (lc._geoBandOverlays) Object.values(lc._geoBandOverlays).forEach(apply);
                apply(lc.currentOverlay);
            }

            // Band-registration mode: filter the CRS.Simple map's overlay panes too — fine
            // there because that map only ever holds local rasters.
            const localMapEl = document.getElementById('local-map');
            if (localMapEl) {
                const panes = localMapEl.querySelectorAll('.leaflet-overlay-pane, .leaflet-tile-pane');
                panes.forEach(p => { p.style.filter = cssFilter; });
            }
        }

        // Back-compat alias for any caller of the old name.
        _applyRenderingToLocalMap() {
            this._applyRenderingToLocalOverlays();
        }

        _currentBands() {
            // Slots are global IDs (#s2-slot-r etc.) so a document-level lookup works
            // regardless of where they're nested.
            const r = document.querySelector(`#${this.prefix}-slot-r .slot-band`)?.dataset.band;
            const g = document.querySelector(`#${this.prefix}-slot-g .slot-band`)?.dataset.band;
            const b = document.querySelector(`#${this.prefix}-slot-b .slot-band`)?.dataset.band;
            const fallback = this.prefix === 's1'
                ? ['VV', 'VH', 'VV']
                : (this.prefix === 's2' ? ['B4', 'B3', 'B2'] : []);
            const bands = [r, g, b].filter(Boolean);
            return bands.length === 3 ? bands : fallback;
        }

        _getAOI() {
            const mm = window.mapManager;
            if (!mm) return null;
            const bbox = (typeof mm.getCurrentBounds === 'function') ? mm.getCurrentBounds() : null;
            const geom = (typeof mm.getCurrentGeoJSON === 'function')
                ? (mm.getCurrentGeoJSON()?.geometry || null)
                : null;
            if (!bbox) return null;
            return { bbox, geometry: geom };
        }

        _notify(msg, level, ms) {
            if (window.app && typeof window.app.showNotification === 'function') {
                window.app.showNotification(msg, level || 'info', ms || 3000);
            } else {
                console.log(`[Symbology:${this.prefix}] ${msg}`);
            }
        }

        // ---------- Public API ----------

        /**
         * Returns the request-body patch to merge into a /get-gee-tile or /get-s{1,2}-tile call.
         * In minmax mode, sends 3-element `min`/`max` arrays (R,G,B order). In percentile mode,
         * sends `stretch_mode` + percentile params; backend computes per-band ranges.
         */
        getRequestPatch() {
            if (this.state.mode === 'percentile') {
                return {
                    stretch_mode: 'percentile',
                    pct_low: this.state.pctLow,
                    pct_high: this.state.pctHigh,
                };
            }
            return {
                stretch_mode: 'minmax',
                min: [this.state.bands.r.min, this.state.bands.g.min, this.state.bands.b.min],
                max: [this.state.bands.r.max, this.state.bands.g.max, this.state.bands.b.max],
            };
        }

        getRenderingState() {
            return { ...this.state.rendering };
        }
    }

    function init() {
        window.symbology = window.symbology || {};
        ['s2', 's1', 'local'].forEach(prefix => {
            const block = document.querySelector(`.symbology-block[data-prefix="${prefix}"]`);
            if (block && !window.symbology[prefix]) {
                window.symbology[prefix] = new SymbologyController(prefix);
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.SymbologyController = SymbologyController;
})();
