/**
 * Spectral Inspector Controller
 *
 * Multi-spectrum accumulation tool: click points or draw areas to collect
 * spectral profiles. Each entry gets a unique color and shows the spectrum
 * as a line chart plus overlay layer values as compact badges.
 *
 * Works identically for satellite search and local image (load-image) modes.
 */

class SpectralInspectorController {
    constructor(platformController) {
        this.platform = platformController;
        this.activeMode = null; // 'point' | null
        this.markers = L.layerGroup();

        // Accumulation state
        this.entries = [];
        this.colorPalette = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990'
        ];
        this.nextColorIdx = 0;
        this.entryCharts = {};
        this.combinedChart = null;

        this._boundPointClick = this._handlePointClick.bind(this);

        // Per-slot stash: see /home/hjh1037/.claude/plans/2-quiet-metcalfe.md §1.5
        this._slotStash = { A: null, B: null };

        this.init();
    }

    init() {
        const map = window.mapManager?.map;
        if (map) {
            if (!map.getPane('spectralPane')) {
                map.createPane('spectralPane');
                map.getPane('spectralPane').style.zIndex = 550;
            }
            this.markers.addTo(map);
        }

        document.getElementById('spectral-point-tool')?.addEventListener('click', () => this._togglePointTool());
        document.getElementById('spectral-clear-btn')?.addEventListener('click', () => this.clearAll());
    }

    // ===== Mode Detection (unified) =====

    _isUploadedMode() {
        const li = this.platform.localImage;
        return li?.currentMode === 'load-image' && !!li?.isUploadedImageActive;
    }

    _isLocalMode() {
        const lc = this.platform.localImage;
        return lc?.localMapActive && lc.currentImageDir && lc.currentAlgoDir;
    }

    // ===== Color =====

    _nextColor() {
        return this.colorPalette[this.nextColorIdx++ % this.colorPalette.length];
    }

    // ===== Toggle Tools =====

    _togglePointTool() {
        if (this.activeMode === 'point') {
            this.deactivate();
        } else {
            this._activatePointTool();
        }
    }

    // ===== Activate Tools =====

    _activatePointTool() {
        this.deactivate();
        this.activeMode = 'point';
        this._setButtonActive('spectral-point-tool');
        this._setStatus('Click on the map to inspect a pixel.');

        const map = window.mapManager?.map;
        if (map) {
            map.getContainer().classList.add('spectral-cursor');
            map.on('click', this._boundPointClick);
        }
    }

    deactivate() {
        const map = window.mapManager?.map;
        if (map) {
            map.off('click', this._boundPointClick);
            map.getContainer().classList.remove('spectral-cursor');
        }
        this.activeMode = null;
        document.getElementById('spectral-point-tool')?.classList.remove('active');
    }

    clearAll() {
        this.deactivate();
        this.markers.clearLayers();
        this.entries = [];
        this.nextColorIdx = 0;

        Object.values(this.entryCharts).forEach(c => c.destroy());
        this.entryCharts = {};
        if (this.combinedChart) { this.combinedChart.destroy(); this.combinedChart = null; }

        const container = document.getElementById('spectral-entries');
        if (container) container.innerHTML = '';
        const combined = document.getElementById('spectral-combined-container');
        if (combined) combined.style.display = 'none';

        this._setStatus('Select a tool and click on the map.');
    }

    // ========== Per-slot stash (Time A / Time B) ==========

    _freezeToSlot(slotId) {
        if (slotId !== 'A' && slotId !== 'B') return;
        // Unbind any active click/draw tool before we detach from the map.
        this.deactivate();

        // Detach the layerGroup wholesale — point markers + area rectangles
        // inside remain attached to it and are preserved.
        const map = window.mapManager?.map;
        if (map && this.markers) {
            try { this.markers.removeFrom(map); } catch (e) { /* no-op */ }
        }

        // Chart.js instances bind to specific <canvas> elements; since
        // _renderEntries will recreate those canvases on thaw, destroy the
        // current instances here to release handles cleanly.
        Object.values(this.entryCharts || {}).forEach(c => {
            try { c.destroy(); } catch (e) { /* no-op */ }
        });
        if (this.combinedChart) {
            try { this.combinedChart.destroy(); } catch (e) { /* no-op */ }
        }

        this._slotStash[slotId] = {
            entries: this.entries.slice(),
            nextColorIdx: this.nextColorIdx,
            markers: this.markers,   // L.layerGroup (detached from map above)
        };

        // Reset live fields — _thawFromSlot will refill them.
        this.entryCharts = {};
        this.combinedChart = null;
    }

    _thawFromSlot(slotId) {
        const stash = (slotId === 'A' || slotId === 'B') ? this._slotStash[slotId] : null;
        if (stash) {
            this.entries = Array.isArray(stash.entries) ? stash.entries.slice() : [];
            this.nextColorIdx = stash.nextColorIdx || 0;
            this.markers = stash.markers || L.layerGroup();
            this._slotStash[slotId] = null;
        } else {
            this.entries = [];
            this.nextColorIdx = 0;
            this.markers = L.layerGroup();
        }
        const map = window.mapManager?.map;
        if (map && this.markers) {
            try { this.markers.addTo(map); } catch (e) { /* no-op */ }
        }
    }

    /**
     * Swap inspector state between slots. `#spectral-entries` always lives in
     * its own right-panel tab, so we always re-render; that recreates the
     * per-entry <canvas> elements and rebinds Chart.js.
     */
    handleSlotChange(oldSlot, newSlot) {
        if (oldSlot === newSlot) return;
        this._freezeToSlot(oldSlot);
        this._thawFromSlot(newSlot);
        if (typeof this._renderEntries === 'function') this._renderEntries();
        if (typeof this._updateCombinedChart === 'function') this._updateCombinedChart();
        if (typeof this._setStatus === 'function') {
            this._setStatus(this.entries.length === 0
                ? 'Select a tool and click on the map.'
                : `${this.entries.length} entr${this.entries.length > 1 ? 'ies' : 'y'} collected.`);
        }
    }

    // ===== Point Click =====

    async _handlePointClick(e) {
        // Ignore clicks from other Leaflet interactions (e.g. draw)
        if (this.activeMode !== 'point') return;

        const { lat, lng } = e.latlng;
        const color = this._nextColor();

        // Diamond-shaped marker (distinct from target detection circles)
        const marker = L.marker([lat, lng], {
            icon: L.divIcon({
                className: 'spectral-diamond-marker',
                html: `<div style="width:12px;height:12px;background:${color};border:2px solid #fff;transform:rotate(45deg);box-shadow:0 0 3px rgba(0,0,0,0.5);"></div>`,
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            }),
            pane: 'spectralPane'
        });
        this.markers.addLayer(marker);

        this._setStatus(`Fetching values at (${lat.toFixed(5)}, ${lng.toFixed(5)})...`);

        try {
            const [bandsData, overlayValues] = await Promise.all([
                this._getSpectralValues(lat, lng),
                this._getOverlayValues(lat, lng)
            ]);

            if (bandsData?.bands?.length > 0) {
                this.entries.push({
                    id: Date.now(),
                    type: 'point',
                    color,
                    latlng: { lat, lng },
                    bands: bandsData.bands,
                    stdBands: null,
                    overlayValues,
                    marker
                });
                this._renderEntries();
                this._updateCombinedChart();
                this._setStatus(`Point: (${lat.toFixed(5)}, ${lng.toFixed(5)}) — ${bandsData.bands.length} bands`);
            } else {
                this.markers.removeLayer(marker);
                this._setStatus(bandsData?.error || 'No data at this location.');
            }
        } catch (err) {
            console.error('[SpectralInspector] Point click error:', err);
            this.markers.removeLayer(marker);
            this._setStatus('Error fetching spectral data.');
        }
    }

    // ===== API =====

    async _getSpectralValues(lat, lng) {
        const localCtrl = this.platform.localImage;

        // Band registration (local CRS.Simple map) mode
        if (this._isLocalMode()) {
            const row = Math.round(lat);
            const col = Math.round(lng);
            try {
                const res = await fetch('/api/local/get-pixel-values', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_dir: localCtrl.currentImageDir,
                        algorithm_dir: localCtrl.currentAlgoDir,
                        row, col
                    })
                });
                return res.ok ? await res.json() : null;
            } catch { return null; }
        }

        // Uploaded GeoTIFF (load-image) mode
        if (this._isUploadedMode()) {
            const px = localCtrl.latlngToPixel({ lat, lng });
            if (!px) return { error: 'Outside image bounds.' };
            try {
                const res = await fetch('/api/local/uploaded/get-pixel-values', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        upload_id: localCtrl.uploadId,
                        row: px.row, col: px.col
                    })
                });
                return res.ok ? await res.json() : null;
            } catch { return null; }
        }

        // Satellite image mode
        const imageId = this.platform.selectedImageId || this.platform.currentProcessedImageId;
        const bbox = window.mapManager?.getCurrentBounds();
        if (!imageId || !bbox) return { error: 'No processed image available.' };

        try {
            const res = await fetch('/api/get-spectral-values', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_id: imageId, bbox, lat, lng })
            });
            return res.ok ? await res.json() : null;
        } catch { return null; }
    }

    async _getOverlayValues(lat, lng) {
        // Pull analysis layers from BOTH slots + any live overlays that
        // weren't registered through the panel (e.g. local CRS.Simple mode).
        // Behaviour is the same for satellite search and local image modes —
        // we just query whichever layers currently exist.
        const layerPanel = this.platform.layerControlPanel;
        const bySlot = layerPanel?.getAllAnalysisLayersBySlot?.() || { A: [], B: [] };
        const crossSlotIds = [...bySlot.A, ...bySlot.B]
            .filter(l => l.type === 'analysis')
            .map(l => l.id);

        const analysisLayers = window.mapManager?.analysisLayers || {};
        const allModelIds = new Set([
            ...crossSlotIds,
            ...Object.keys(analysisLayers),
        ]);

        // Exclude base images, tiles, masks, segmentation
        const excluded = id =>
            id === 'model1' ||
            id === 'local-base-image' ||
            id.startsWith('preview-') ||
            id.includes('mangrove-seg');

        const validIds = [...allModelIds].filter(id => !excluded(id));
        if (validIds.length === 0) return [];

        const imageId = this.platform.selectedImageId || this.platform.currentProcessedImageId;
        const bbox = window.mapManager?.getCurrentBounds();
        const localCtrl = this.platform.localImage;

        const labelMap = {
            'model2': 'NDVI', 'model3': 'NDMI', 'model4': 'MVI', 'model5': 'AlphaEarth', 'cloud_mask': 'Cloud'
        };

        const promises = validIds.map(async (modelId) => {
            try {
                let res;

                // For uploaded mode, use the uploaded pixel endpoint
                if (this._isUploadedMode() && localCtrl?.uploadId) {
                    const px = localCtrl.latlngToPixel({ lat, lng });
                    if (!px) return null;
                    res = await fetch('/api/local/uploaded/get-index-value', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ upload_id: localCtrl.uploadId, model_id: modelId, row: px.row, col: px.col })
                    });
                } else {
                    // Satellite mode
                    res = await fetch('/api/get-pixel-value', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_id: imageId || '', lat, lng, model_id: modelId, bbox: bbox || [] })
                    });
                }

                if (!res.ok) return null;
                const data = await res.json();
                if (data.error || typeof data.value === 'string') return null;

                // Generate readable label
                let label = labelMap[modelId] || modelId;
                if (modelId.startsWith('td-')) label = 'TD';
                else if (modelId.startsWith('sam3-')) label = 'SAM3';
                else if (modelId.startsWith('spectral-')) label = 'Index';
                else if (modelId.startsWith('custom-result-')) label = 'Custom';

                return { modelId, label, value: data.value };
            } catch { return null; }
        });

        const results = await Promise.all(promises);
        return results.filter(Boolean);
    }

    // ===== Rendering =====

    _renderEntries() {
        const container = document.getElementById('spectral-entries');
        if (!container) return;

        Object.values(this.entryCharts).forEach(c => c.destroy());
        this.entryCharts = {};
        container.innerHTML = '';

        // Render newest first
        for (let i = this.entries.length - 1; i >= 0; i--) {
            const entry = this.entries[i];
            const card = document.createElement('div');
            card.className = 'spectral-entry-card';
            card.style.borderLeft = `4px solid ${entry.color}`;

            const typeIcon = '◆';
            const coordText = `(${entry.latlng.lat.toFixed(4)}, ${entry.latlng.lng.toFixed(4)})`;

            // Build layer tabs
            const tabs = this._buildLayerTabs(entry);
            const baseCanvasId = `spectral-entry-chart-${entry.id}`;

            card.innerHTML = `
                <div class="entry-header">
                    <div class="entry-label">
                        <span class="entry-color-dot" style="background:${entry.color}"></span>
                        <span>${typeIcon} ${coordText}</span>
                    </div>
                    <button class="entry-remove" data-entry-id="${entry.id}" title="Remove">✕</button>
                </div>
                ${tabs.tabBar}
                <div class="spectral-tab-content" data-entry-id="${entry.id}">
                    <div class="spectral-tab-pane active" data-tab="base">
                        <div class="entry-chart"><canvas id="${baseCanvasId}"></canvas></div>
                    </div>
                    ${tabs.panes}
                </div>
            `;

            card.querySelector('.entry-remove').addEventListener('click', (e) => {
                e.stopPropagation();
                this._removeEntry(entry.id);
            });

            // Tab click handlers
            card.querySelectorAll('.spectral-layer-tab').forEach(tabBtn => {
                tabBtn.addEventListener('click', () => {
                    const targetTab = tabBtn.dataset.tab;
                    const tabContent = card.querySelector(`.spectral-tab-content[data-entry-id="${entry.id}"]`);
                    tabContent.querySelectorAll('.spectral-tab-pane').forEach(p => p.classList.remove('active'));
                    tabContent.querySelector(`.spectral-tab-pane[data-tab="${targetTab}"]`)?.classList.add('active');
                    card.querySelectorAll('.spectral-layer-tab').forEach(t => t.classList.remove('active'));
                    tabBtn.classList.add('active');
                });
            });

            container.appendChild(card);
            requestAnimationFrame(() => this._createEntryChart(entry, baseCanvasId));
        }
    }

    _buildLayerTabs(entry) {
        const overlays = entry.overlayValues || [];
        if (overlays.length === 0) {
            return { tabBar: '', panes: '' };
        }

        let tabButtons = `<button class="spectral-layer-tab active" data-tab="base">Base Image</button>`;
        let panes = '';

        overlays.forEach((ov, idx) => {
            const tabId = `overlay-${idx}`;
            tabButtons += `<button class="spectral-layer-tab" data-tab="${tabId}">${ov.label}</button>`;
            const valDisplay = typeof ov.value === 'number' ? ov.value.toFixed(4) : ov.value;
            panes += `
                <div class="spectral-tab-pane" data-tab="${tabId}">
                    <div class="spectral-overlay-value">
                        <span class="overlay-value-label">${ov.label}</span>
                        <span class="overlay-value-number">${valDisplay}</span>
                        <span class="overlay-value-id">${ov.modelId}</span>
                    </div>
                </div>
            `;
        });

        const tabBar = `<div class="spectral-layer-tabs">${tabButtons}</div>`;
        return { tabBar, panes };
    }

    _createEntryChart(entry, canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const labels = entry.bands.map(b => b.name);
        const values = entry.bands.map(b => b.value);
        const datasets = [];

        if (entry.type === 'area' && entry.stdBands) {
            const stdValues = entry.stdBands.map(b => b.value);
            const upper = values.map((v, i) => v + stdValues[i]);
            const lower = values.map((v, i) => v - stdValues[i]);

            datasets.push({
                label: '+1σ',
                data: upper,
                borderColor: 'transparent',
                backgroundColor: entry.color + '30',
                pointRadius: 0,
                fill: '+1',
                tension: 0.3
            });
            datasets.push({
                label: '-1σ',
                data: lower,
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                pointRadius: 0,
                fill: false,
                tension: 0.3
            });
        }

        datasets.push({
            label: entry.type === 'area' ? 'Mean' : 'Spectrum',
            data: values,
            borderColor: entry.color,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 2,
            pointBackgroundColor: entry.color,
            tension: 0.3,
            fill: false
        });

        this.entryCharts[entry.id] = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                plugins: {
                    legend: { display: false },
                    filler: { propagate: true }
                },
                scales: {
                    y: { beginAtZero: false, ticks: { font: { size: 9 } } },
                    x: { ticks: { font: { size: 9 }, maxRotation: 45 } }
                }
            }
        });
    }

    // ===== Combined Chart =====

    _updateCombinedChart() {
        const container = document.getElementById('spectral-combined-container');
        const canvas = document.getElementById('spectral-combined-chart');
        if (!container || !canvas) return;

        if (this.entries.length < 2) {
            container.style.display = 'none';
            if (this.combinedChart) { this.combinedChart.destroy(); this.combinedChart = null; }
            return;
        }

        container.style.display = 'block';
        if (this.combinedChart) this.combinedChart.destroy();

        const labels = this.entries[0].bands.map(b => b.name);
        const datasets = this.entries.map((entry, idx) => ({
            label: entry.type === 'point' ? `P${idx + 1}` : `A${idx + 1}`,
            data: entry.bands.map(b => b.value),
            borderColor: entry.color,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 1,
            tension: 0.3,
            fill: false
        }));

        this.combinedChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { boxWidth: 12, font: { size: 9 }, padding: 6 }
                    }
                },
                scales: {
                    y: { beginAtZero: false, ticks: { font: { size: 9 } } },
                    x: { ticks: { font: { size: 9 }, maxRotation: 45 } }
                }
            }
        });
    }

    // ===== Remove Entry =====

    _removeEntry(entryId) {
        const idx = this.entries.findIndex(e => e.id === entryId);
        if (idx === -1) return;

        const entry = this.entries[idx];
        if (entry.marker) this.markers.removeLayer(entry.marker);
        if (entry.mapLayer) this.markers.removeLayer(entry.mapLayer);

        if (this.entryCharts[entryId]) {
            this.entryCharts[entryId].destroy();
            delete this.entryCharts[entryId];
        }

        this.entries.splice(idx, 1);
        this._renderEntries();
        this._updateCombinedChart();

        if (this.entries.length === 0) {
            this._setStatus('Select a tool and click on the map.');
        }
    }

    // ===== UI Helpers =====

    _setStatus(msg) {
        const el = document.getElementById('spectral-status');
        if (el) el.textContent = msg;
    }

    _setButtonActive(id) {
        document.getElementById('spectral-point-tool')?.classList.remove('active');
        document.getElementById('spectral-area-tool')?.classList.remove('active');
        document.getElementById(id)?.classList.add('active');
    }
}
