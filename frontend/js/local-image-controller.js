/**
 * Local Image Controller
 *
 * Two modes:
 * 1. Load Image — user uploads a GeoTIFF with CRS, displayed on main Leaflet map with basemap.
 *    Analysis (spectral, target detection) is enabled.
 * 2. Band Registration — browse experiment_results directory, CRS.Simple map (no basemap).
 *    Analysis is disabled.
 */

class LocalImageController {
    constructor(platformController) {
        this.platform = platformController;

        // Mode: 'load-image' or 'band-registration'
        this.currentMode = 'load-image';

        // Band Registration state (existing)
        this.currentImageDir = null;
        this.currentAlgoDir = null;
        this.availableBands = [];
        this.currentOverlay = null;
        this.localMap = null;
        this.localMapActive = false;
        this.preloadedBandUrls = {};
        this.preloadedSize = { width: 0, height: 0 };
        this.percentiles = null;
        this.keyboardShortcutMode = false;
        this.keyboardShortcutMap = {};
        this.localAnalysisOverlays = {};
        this.lastLocalAnalysisResults = null;

        // Load Image state (new)
        this.uploadedImageMeta = null;
        this.uploadId = null;
        this.geoOverlay = null;       // L.imageOverlay on main map
        this.geoBounds = null;        // [[south, west], [north, east]]
        this.geoAnalysisOverlays = {}; // modelId: L.imageOverlay on main map
        this.isUploadedImageActive = false;
        this.uploadedBandRoles = {};   // {"NIR": 4, "RED": 3, ...} 1-based

        // Per-slot stash for Load-Image state. Each slot holds the full
        // bundle of upload-related fields so switching slots swaps out the
        // overlay, metadata UI, band pool etc. all at once. Band-
        // registration mode state is intentionally NOT slot-scoped — it's a
        // separate workflow that operates on a secondary CRS.Simple map.
        this._localSlotState = { A: null, B: null };

        // Per-channel cache for true per-band Min/Max compositing in load-image
        // mode. Keys are 'r'/'g'/'b'; each holds the loaded HTMLImageElement plus
        // the (band, min, max) tuple it was stretched for, so we can avoid
        // refetching when only an unrelated channel changed.
        this._slotImages = { r: null, g: null, b: null };
        this._slotMeta   = { r: null, g: null, b: null };

        this.init();
    }

    _dispatchLayerEvent(eventName, detail) {
        try {
            window.dispatchEvent(new CustomEvent(eventName, { detail }));
        } catch (e) {
            console.warn(`[LocalImage] Event dispatch error (${eventName}):`, e);
        }
    }

    // ========== Per-slot state management ==========

    /**
     * Snapshot all Load-Image-mode fields into `_localSlotState[slotId]` so
     * they can be restored when the user flips back. Leaflet overlay objects
     * are kept alive (they're just removed from the map, not destroyed), so
     * switching back shows the image instantly without re-uploading.
     */
    _freezeToSlot(slotId) {
        if (slotId !== 'A' && slotId !== 'B') return;
        // Also capture DOM-only state that doesn't live on the controller:
        // the RGB slot band assignments and the min/max stretch inputs. The
        // user's picks here should swap with the slot, not leak across.
        const rgbSlots = {};
        ['r', 'g', 'b'].forEach(ch => {
            const slotBand = document.querySelector(`#local-slot-${ch} .slot-band`);
            rgbSlots[ch] = slotBand?.dataset?.band || null;
        });
        const minEl = document.getElementById('local-min');
        const maxEl = document.getElementById('local-max');

        this._localSlotState[slotId] = {
            uploadId: this.uploadId,
            uploadedImageMeta: this.uploadedImageMeta,
            geoOverlay: this.geoOverlay,
            geoBounds: this.geoBounds,
            geoAnalysisOverlays: { ...(this.geoAnalysisOverlays || {}) },
            _geoBandOverlays: { ...(this._geoBandOverlays || {}) },
            _bandImages: { ...(this._bandImages || {}) },
            isUploadedImageActive: this.isUploadedImageActive,
            uploadedBandRoles: { ...(this.uploadedBandRoles || {}) },
            availableBands: Array.isArray(this.availableBands) ? this.availableBands.slice() : [],
            preloadedBandUrls: { ...(this.preloadedBandUrls || {}) },
            preloadedSize: { ...(this.preloadedSize || { width: 0, height: 0 }) },
            percentiles: this.percentiles,
            _preloadedMin: this._preloadedMin,
            _preloadedMax: this._preloadedMax,
            _slotImages: { ...(this._slotImages || { r: null, g: null, b: null }) },
            _slotMeta:   { ...(this._slotMeta   || { r: null, g: null, b: null }) },
            lastLocalAnalysisResults: this.lastLocalAnalysisResults,
            rgbSlots,
            minValue: minEl?.value ?? null,
            maxValue: maxEl?.value ?? null,
        };
    }

    /**
     * Load the previously-stashed state for `slotId` into the live fields.
     * Resets to empty if the slot has never been populated.
     */
    _thawFromSlot(slotId) {
        const s = (slotId === 'A' || slotId === 'B') ? this._localSlotState[slotId] : null;
        if (s) {
            this.uploadId = s.uploadId || null;
            this.uploadedImageMeta = s.uploadedImageMeta || null;
            this.geoOverlay = s.geoOverlay || null;
            this.geoBounds = s.geoBounds || null;
            this.geoAnalysisOverlays = s.geoAnalysisOverlays || {};
            this._geoBandOverlays = s._geoBandOverlays || {};
            this._bandImages = s._bandImages || {};
            this.isUploadedImageActive = !!s.isUploadedImageActive;
            this.uploadedBandRoles = s.uploadedBandRoles || {};
            this.availableBands = s.availableBands || [];
            this.preloadedBandUrls = s.preloadedBandUrls || {};
            this.preloadedSize = s.preloadedSize || { width: 0, height: 0 };
            this.percentiles = s.percentiles || null;
            this._preloadedMin = s._preloadedMin;
            this._preloadedMax = s._preloadedMax;
            this._slotImages = s._slotImages || { r: null, g: null, b: null };
            this._slotMeta   = s._slotMeta   || { r: null, g: null, b: null };
            this.lastLocalAnalysisResults = s.lastLocalAnalysisResults || null;
            // Stash DOM-only state so _refreshUploadUI can apply it after
            // rebuilding the band pool / info panel DOM.
            this._pendingRgbSlots = s.rgbSlots || null;
            this._pendingMinValue = s.minValue;
            this._pendingMaxValue = s.maxValue;
        } else {
            // Slot empty — reset to fresh state.
            this.uploadId = null;
            this.uploadedImageMeta = null;
            this.geoOverlay = null;
            this.geoBounds = null;
            this.geoAnalysisOverlays = {};
            this._geoBandOverlays = {};
            this._bandImages = {};
            this.isUploadedImageActive = false;
            this.uploadedBandRoles = {};
            this.availableBands = [];
            this.preloadedBandUrls = {};
            this.preloadedSize = { width: 0, height: 0 };
            this.percentiles = null;
            this._preloadedMin = undefined;
            this._preloadedMax = undefined;
            this._slotImages = { r: null, g: null, b: null };
            this._slotMeta   = { r: null, g: null, b: null };
            this.lastLocalAnalysisResults = null;
            this._pendingRgbSlots = null;
            this._pendingMinValue = null;
            this._pendingMaxValue = null;
        }
    }

    /**
     * Sync the Load-Image-mode DOM to the currently-live fields. Called
     * after thawing a slot so the file-info panel, band role dropdowns and
     * band pool reflect the new slot.
     */
    _refreshUploadUI() {
        const infoEl = document.getElementById('uploaded-image-info');
        const dropZoneEl = document.getElementById('geotiff-drop-zone');
        const bandCtrls = document.getElementById('local-band-controls');
        const analyzeBtn = document.getElementById('local-open-analysis-btn');

        const hasUpload = !!this.uploadId && !!this.uploadedImageMeta;

        if (hasUpload) {
            const meta = this.uploadedImageMeta;
            const fn = document.getElementById('uploaded-filename');
            const crs = document.getElementById('uploaded-crs');
            const bc = document.getElementById('uploaded-band-count');
            const sz = document.getElementById('uploaded-size');
            if (fn) fn.textContent = meta.filename || '';
            if (crs) crs.textContent = meta.crs || '';
            if (bc) bc.textContent = `${meta.band_count || 0} bands`;
            if (sz) sz.textContent = `${meta.width || 0} x ${meta.height || 0} px`;
            if (infoEl) infoEl.style.display = '';
            if (dropZoneEl) dropZoneEl.style.display = 'none';

            // Rebuild band-role dropdowns from the stored band names, then
            // restore the stored role→band mapping.
            if (Array.isArray(meta.band_names) && meta.band_names.length > 0) {
                this._populateBandRoleMapping(meta.band_names);
                document.querySelectorAll('#band-role-grid select').forEach(sel => {
                    const role = sel.dataset.role;
                    if (this.uploadedBandRoles && this.uploadedBandRoles[role]) {
                        sel.value = String(this.uploadedBandRoles[role]);
                    }
                });
            }

            // Rebuild the draggable band pool from availableBands.
            const pool = document.getElementById('local-band-pool');
            if (pool && Array.isArray(this.availableBands)) {
                pool.innerHTML = '';
                this.availableBands.forEach(band => {
                    const chip = document.createElement('div');
                    chip.className = 'band-chip';
                    chip.draggable = true;
                    chip.dataset.band = band;
                    chip.textContent = band.replace(/^Band\s*/i, '');
                    chip.addEventListener('dragstart', (e) => {
                        e.dataTransfer.setData('text/plain', band);
                        chip.classList.add('dragging');
                    });
                    chip.addEventListener('dragend', () => chip.classList.remove('dragging'));
                    pool.appendChild(chip);
                });
            }

            // Restore RGB slot assignments from the thawed stash (user drops
            // bands onto R/G/B slots — those picks should be per-slot).
            if (this._pendingRgbSlots) {
                ['r', 'g', 'b'].forEach(ch => {
                    const slotBand = document.querySelector(`#local-slot-${ch} .slot-band`);
                    if (!slotBand) return;
                    const band = this._pendingRgbSlots[ch];
                    if (band) {
                        slotBand.textContent = String(band).replace(/^Band\s*/i, '');
                        slotBand.dataset.band = band;
                    } else {
                        slotBand.textContent = '';
                        delete slotBand.dataset.band;
                    }
                });
                this._pendingRgbSlots = null;
            }
            // Restore min/max stretch inputs.
            if (this._pendingMinValue != null) {
                const minEl = document.getElementById('local-min');
                if (minEl) minEl.value = String(this._pendingMinValue);
                this._pendingMinValue = null;
            }
            if (this._pendingMaxValue != null) {
                const maxEl = document.getElementById('local-max');
                if (maxEl) maxEl.value = String(this._pendingMaxValue);
                this._pendingMaxValue = null;
            }

            if (bandCtrls) bandCtrls.style.display = '';
            if (analyzeBtn) analyzeBtn.disabled = false;
        } else {
            if (infoEl) infoEl.style.display = 'none';
            if (dropZoneEl) dropZoneEl.style.display = '';
            if (bandCtrls) bandCtrls.style.display = 'none';
            if (analyzeBtn) analyzeBtn.disabled = true;

            const grid = document.getElementById('band-role-grid');
            if (grid) grid.innerHTML = '';
            const pool = document.getElementById('local-band-pool');
            if (pool) pool.innerHTML = '';

            // Clear RGB slot bands — leave visual slots but remove any band
            // dataset so nothing accidentally "carries over" into an empty slot.
            ['r', 'g', 'b'].forEach(ch => {
                const slotBand = document.querySelector(`#local-slot-${ch} .slot-band`);
                if (!slotBand) return;
                slotBand.textContent = '';
                delete slotBand.dataset.band;
            });
        }
    }

    /**
     * Re-attach all stashed Leaflet overlays for the active slot to the main
     * map. Called after `_thawFromSlot` when we're in 'load-image' mode.
     */
    _restoreGeoOverlaysToMap() {
        const map = window.mapManager?.map;
        if (!map) return;
        if (this.geoOverlay && !map.hasLayer(this.geoOverlay)) {
            this.geoOverlay.addTo(map);
        }
        if (this._geoBandOverlays) {
            Object.values(this._geoBandOverlays).forEach(ov => {
                if (ov && !map.hasLayer(ov)) {
                    ov.addTo(map);
                    // Preserve the original display:none convention used by
                    // _loadGeoImages (only the active RGB composite is shown).
                    const el = ov.getElement && ov.getElement();
                    if (el) el.style.display = 'none';
                }
            });
        }
        // geoAnalysisOverlays are already managed by MapLayers' slot stash,
        // so we don't re-add them here to avoid double-adding.
    }

    /**
     * Return the filename of the uploaded GeoTIFF for `slotId`, or null if
     * that slot has no upload. Used by PlatformController._updateSlotToolbar
     * to show a "local: <name>" hint in the slot button subtitle.
     */
    getSlotUploadFilename(slotId) {
        const activeSlot = this.platform?.currentSlot || 'A';
        if (slotId === activeSlot) {
            return this.uploadedImageMeta?.filename || null;
        }
        return this._localSlotState?.[slotId]?.uploadedImageMeta?.filename || null;
    }

    /**
     * Called by PlatformController.switchSlot. Swap out the live state for
     * `oldSlot` and swap in the state for `newSlot`, including map overlays
     * (when currently in Load-Image mode) and DOM.
     */
    handleSlotChange(oldSlot, newSlot) {
        if (oldSlot === newSlot) return;

        // 1. Stash outgoing slot.
        this._freezeToSlot(oldSlot);

        // 2. Strip current overlays off the map. Band overlays belong to the
        //    outgoing slot; pulling them here prevents two slots worth of
        //    imagery showing at once while we thaw the incoming slot.
        this._removeGeoOverlays();

        // 3. Load incoming slot's fields.
        this._thawFromSlot(newSlot);

        // 4. Refresh DOM (drop zone / info panel / band controls / analyze
        //    button) regardless of current mode — the changes are harmless
        //    even when the Load-Image panel isn't visible right now.
        this._refreshUploadUI();

        // 5. Put overlays back on the map only if the user is actually
        //    looking at the main map in Load-Image mode. In band-reg mode
        //    we leave them detached until the mode toggle runs them back on.
        if (this.currentMode === 'load-image') {
            this._restoreGeoOverlaysToMap();
        }

        // 6. If this slot had a local Analyze session (the user clicked the
        //    local "Analyze" button before switching away), rebuild the
        //    right-panel `.analysis-list` with local options. Without this,
        //    `PlatformController.switchSlot` has already called
        //    `imageProcessorController.showAnalysisResults(null)` which paints
        //    the satellite empty-state and hides the local entry points.
        if (this.isUploadedImageActive && this.lastLocalAnalysisResults != null) {
            this._renderLocalAnalysisListDOM();
        }
    }

    init() {
        this.setupModeToggle();
        this.setupSelectors();
        this.setupApplyButton();
        this.setupDragDrop();
        this.setupKeyboardShortcut();
        this.setupTabWatcher();
        this.setupPercentileSlider();
        this.setupFileUpload();
        this.loadImages();
    }

    // ===== Mode Toggle =====

    setupModeToggle() {
        document.querySelectorAll('.local-mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                if (mode === this.currentMode) return;

                document.querySelectorAll('.local-mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentMode = mode;

                const loadPanel = document.getElementById('load-image-mode');
                const bandRegPanel = document.getElementById('band-registration-mode');
                const analyzeBtn = document.getElementById('local-open-analysis-btn');

                if (mode === 'load-image') {
                    loadPanel.style.display = '';
                    bandRegPanel.style.display = 'none';
                    // Hide CRS.Simple map, show main map
                    this.hideLocalMap();
                    // Show uploaded image overlay if exists
                    if (this.uploadId && this.geoOverlay) {
                        this._showMainMap();
                        if (!window.mapManager.map.hasLayer(this.geoOverlay)) {
                            this.geoOverlay.addTo(window.mapManager.map);
                        }
                    }
                    // Enable analyze if image loaded
                    if (analyzeBtn) analyzeBtn.disabled = !this.uploadId;
                    // Show band controls if bands loaded
                    if (this.uploadId && this.availableBands.length > 0) {
                        document.getElementById('local-band-controls').style.display = '';
                    }
                } else {
                    loadPanel.style.display = 'none';
                    bandRegPanel.style.display = '';
                    // Hide uploaded overlay from main map
                    this._removeGeoOverlays();
                    // Show CRS.Simple map if we have band-reg data
                    if (this.currentImageDir && this.currentAlgoDir) {
                        this.showLocalMap();
                    }
                    // Disable analyze in band registration mode
                    if (analyzeBtn) analyzeBtn.disabled = true;
                    // Show band controls if algo selected
                    if (this.currentAlgoDir && this._bandRegBandsLoaded) {
                        document.getElementById('local-band-controls').style.display = '';
                    } else {
                        document.getElementById('local-band-controls').style.display = 'none';
                    }
                }
            });
        });
    }

    _showMainMap() {
        // Ensure main map is visible
        if (window.mapManager?.map) {
            window.mapManager.map.getContainer().style.display = '';
            window.mapManager.map.invalidateSize();
        }
        const localMapEl = document.getElementById('local-map');
        if (localMapEl) localMapEl.style.display = 'none';
    }

    _removeGeoOverlays() {
        const map = window.mapManager?.map;
        if (!map) return;
        if (this.geoOverlay && map.hasLayer(this.geoOverlay)) {
            map.removeLayer(this.geoOverlay);
        }
        // Remove band overlays from main map
        if (this._geoBandOverlays) {
            Object.values(this._geoBandOverlays).forEach(ov => {
                if (map.hasLayer(ov)) map.removeLayer(ov);
            });
        }
        // Remove analysis overlays from main map
        Object.values(this.geoAnalysisOverlays).forEach(ov => {
            if (map.hasLayer(ov)) map.removeLayer(ov);
        });
    }

    // ===== File Upload (Load Image mode) =====

    setupFileUpload() {
        const dropZone = document.getElementById('geotiff-drop-zone');
        const fileInput = document.getElementById('geotiff-file-input');
        const browseLink = document.getElementById('geotiff-browse-link');
        const removeBtn = document.getElementById('remove-uploaded-image');

        if (!dropZone || !fileInput) return;

        browseLink?.addEventListener('click', (e) => {
            e.preventDefault();
            fileInput.click();
        });

        dropZone.addEventListener('click', (e) => {
            if (e.target === browseLink) return;
            fileInput.click();
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) this.uploadGeoTIFF(file);
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) this.uploadGeoTIFF(file);
            fileInput.value = '';
        });

        removeBtn?.addEventListener('click', () => this.removeUploadedImage());
    }

    async uploadGeoTIFF(file) {
        this.platform.showLoading('Uploading GeoTIFF...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/api/local/upload-geotiff', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const meta = await res.json();
            this.uploadedImageMeta = meta;
            this.uploadId = meta.upload_id;
            this.isUploadedImageActive = true;

            // Set geo bounds: [[south, west], [north, east]]
            const [west, south, east, north] = meta.bounds_4326;
            this.geoBounds = [[south, west], [north, east]];

            // Display metadata
            document.getElementById('uploaded-filename').textContent = meta.filename;
            document.getElementById('uploaded-crs').textContent = meta.crs;
            document.getElementById('uploaded-band-count').textContent = `${meta.band_count} bands`;
            document.getElementById('uploaded-size').textContent = `${meta.width} x ${meta.height} px`;
            document.getElementById('uploaded-image-info').style.display = '';
            document.getElementById('geotiff-drop-zone').style.display = 'none';

            // Populate band role mapping
            this._populateBandRoleMapping(meta.band_names);

            // Load bands via GPU
            await this.loadUploadedBands();

            // Refresh the slot toolbar so the Time A / Time B buttons show
            // the uploaded filename for the active slot.
            if (typeof this.platform._updateSlotToolbar === 'function') {
                this.platform._updateSlotToolbar();
            }

            this.platform.showNotification('GeoTIFF uploaded successfully', 'success');
        } catch (e) {
            console.error('Upload failed:', e);
            this.platform.showNotification(`Upload failed: ${e.message}`, 'error');
            this.platform.hideLoading();
        }
    }

    _populateBandRoleMapping(bandNames) {
        const grid = document.getElementById('band-role-grid');
        if (!grid) return;
        grid.innerHTML = '';

        const roles = ['NIR', 'RED', 'GREEN', 'BLUE', 'SWIR', 'REDEDGE'];
        const optionsHtml = bandNames.map((name, i) =>
            `<option value="${i + 1}">${name.replace(/^Band\s*/i, 'Band ')}</option>`
        ).join('');

        roles.forEach(role => {
            const item = document.createElement('div');
            item.className = 'band-role-item';
            item.innerHTML = `
                <label>${role}:</label>
                <select data-role="${role}">
                    <option value="">--</option>
                    ${optionsHtml}
                </select>
            `;
            // Auto-select based on band name matching
            const select = item.querySelector('select');
            bandNames.forEach((name, i) => {
                const upper = name.toUpperCase();
                if (role === 'NIR' && (upper.includes('NIR') || upper.includes('B8') || upper === 'BAND 5' || upper === 'BAND 4'))
                    select.value = i + 1;
                else if (role === 'RED' && (upper.includes('RED') && !upper.includes('EDGE') || upper.includes('B4') || upper === 'BAND 3'))
                    select.value = i + 1;
                else if (role === 'GREEN' && (upper.includes('GREEN') || upper.includes('B3') || upper === 'BAND 2'))
                    select.value = i + 1;
                else if (role === 'BLUE' && (upper.includes('BLUE') || upper.includes('B2') || upper === 'BAND 1'))
                    select.value = i + 1;
                else if (role === 'SWIR' && (upper.includes('SWIR') || upper.includes('B11') || upper === 'BAND 6'))
                    select.value = i + 1;
                else if (role === 'REDEDGE' && (upper.includes('REDEDGE') || upper.includes('RED EDGE') || upper.includes('B5')))
                    select.value = i + 1;
            });

            select.addEventListener('change', () => this._updateBandRoles());
            grid.appendChild(item);
        });

        this._updateBandRoles();
    }

    _updateBandRoles() {
        this.uploadedBandRoles = {};
        document.querySelectorAll('#band-role-grid select').forEach(sel => {
            const role = sel.dataset.role;
            const val = parseInt(sel.value);
            if (val) this.uploadedBandRoles[role] = val;
        });
    }

    async loadUploadedBands() {
        this.platform.showLoading('Loading bands (GPU)...');

        try {
            this.clearOverlay();
            this.preloadedBandUrls = {};
            this.preloadedSize = { width: 0, height: 0 };
            this._bandImages = {};
            this._geoBandOverlays = {};

            const min = parseFloat(document.getElementById('local-min')?.value) || 0;
            const max = parseFloat(document.getElementById('local-max')?.value) || 3000;
            this._preloadedMin = min;
            this._preloadedMax = max;

            const res = await fetch('/api/local/uploaded/gpu-load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    upload_id: this.uploadId,
                    min_val: min,
                    max_val: max
                })
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();

            this.percentiles = data.percentiles;
            this._updateMinMaxFromSliders();
            this.preloadedBandUrls = data.band_urls;
            this.preloadedSize = { width: data.width, height: data.height };

            // Populate band pool
            this.availableBands = data.band_names;
            const pool = document.getElementById('local-band-pool');
            pool.innerHTML = '';
            this.availableBands.forEach(band => {
                const chip = document.createElement('div');
                chip.className = 'band-chip';
                chip.draggable = true;
                chip.dataset.band = band;
                // Shorten label: "Band 1" → "1", keep short names as-is
                chip.textContent = band.replace(/^Band\s*/i, '');
                chip.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', band);
                    chip.classList.add('dragging');
                });
                chip.addEventListener('dragend', () => chip.classList.remove('dragging'));
                pool.appendChild(chip);
            });

            // Set default RGB slots
            const defaults = this.availableBands.slice(0, 3);
            ['local-slot-r', 'local-slot-g', 'local-slot-b'].forEach((slotId, i) => {
                const slotBand = document.querySelector(`#${slotId} .slot-band`);
                if (slotBand && defaults[i]) {
                    slotBand.textContent = defaults[i].replace(/^Band\s*/i, '');
                    slotBand.dataset.band = defaults[i];
                }
            });

            document.getElementById('local-band-controls').style.display = '';
            this.setupShortcutSlots();

            // Enable analyze button
            const analyzeBtn = document.getElementById('local-open-analysis-btn');
            if (analyzeBtn) analyzeBtn.disabled = false;

            // Pre-create Image objects for client-side compositing
            await this._loadGeoImages(data.band_urls, data.width, data.height);

            // Auto-show default RGB
            this._autoShowDefaultRGB();

            this.platform.hideLoading();
            this.platform.showNotification('Bands ready (GPU)', 'success', 1500);
        } catch (e) {
            console.error('loadUploadedBands failed:', e);
            this.platform.hideLoading();
            this.platform.showNotification(`Load failed: ${e.message}`, 'error');
        }
    }

    async _loadGeoImages(bandUrls, width, height) {
        this._bandImages = {};
        this._geoBandOverlays = {};

        const map = window.mapManager?.map;
        if (!map || !this.geoBounds) return;

        // Show main map
        this._showMainMap();

        const loadPromises = Object.entries(bandUrls).map(([band, url]) => {
            return new Promise((resolve) => {
                const img = new window.Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => {
                    this._bandImages[band] = img;
                    const overlay = L.imageOverlay(url, this.geoBounds, { opacity: 1.0, interactive: false });
                    overlay.addTo(map);
                    overlay.getElement().style.display = 'none';
                    this._geoBandOverlays[band] = overlay;
                    resolve();
                };
                img.onerror = () => resolve();
                img.src = url;
            });
        });
        await Promise.all(loadPromises);

        // Fit map to image bounds
        map.fitBounds(this.geoBounds);
    }

    removeUploadedImage() {
        this._removeGeoOverlays();
        this.uploadedImageMeta = null;
        this.uploadId = null;
        this.isUploadedImageActive = false;
        this.geoOverlay = null;
        this.geoBounds = null;
        this._bandImages = {};
        this._geoBandOverlays = {};
        this.geoAnalysisOverlays = {};
        this.preloadedBandUrls = {};
        this.preloadedSize = { width: 0, height: 0 };
        this.availableBands = [];
        this.percentiles = null;
        this.uploadedBandRoles = {};
        this._slotImages = { r: null, g: null, b: null };
        this._slotMeta   = { r: null, g: null, b: null };

        document.getElementById('uploaded-image-info').style.display = 'none';
        document.getElementById('geotiff-drop-zone').style.display = '';
        document.getElementById('local-band-controls').style.display = 'none';
        const analyzeBtn = document.getElementById('local-open-analysis-btn');
        if (analyzeBtn) analyzeBtn.disabled = true;

        if (typeof this.platform._updateSlotToolbar === 'function') {
            this.platform._updateSlotToolbar();
        }

        this.platform.showNotification('Image removed', 'info');
    }

    // ===== Coordinate conversion =====

    latlngToPixel(latlng) {
        if (!this.uploadedImageMeta) return null;
        const [west, south, east, north] = this.uploadedImageMeta.bounds_4326;
        const col = Math.round((latlng.lng - west) / (east - west) * this.uploadedImageMeta.width);
        const row = Math.round((north - latlng.lat) / (north - south) * this.uploadedImageMeta.height);
        return { row, col };
    }

    // ===== GPU re-stretch for uploaded images =====

    async gpuStretchUploaded() {
        const min = parseFloat(document.getElementById('local-min')?.value) || 0;
        const max = parseFloat(document.getElementById('local-max')?.value) || 3000;
        this._preloadedMin = min;
        this._preloadedMax = max;

        try {
            const res = await fetch('/api/local/uploaded/gpu-stretch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    upload_id: this.uploadId,
                    min_val: min,
                    max_val: max
                })
            });
            if (!res.ok) throw new Error('GPU stretch failed');
            const data = await res.json();
            this.preloadedBandUrls = data.band_urls;
            this.preloadedSize = { width: data.width, height: data.height };
            await this._loadGeoImages(data.band_urls, data.width, data.height);
        } catch (e) {
            console.error('gpuStretchUploaded failed:', e);
        }
    }

    // ===== Tab switching: show/hide appropriate map =====

    setupTabWatcher() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.getAttribute('data-tab');
                if (tab === 'local-image') {
                    if (this.currentMode === 'band-registration') {
                        this.showLocalMap();
                    } else {
                        // Load Image mode: show main map
                        this._showMainMap();
                        if (this.geoOverlay && window.mapManager?.map) {
                            if (!window.mapManager.map.hasLayer(this.geoOverlay)) {
                                this.geoOverlay.addTo(window.mapManager.map);
                            }
                        }
                    }
                } else {
                    this.hideLocalMap();
                    // When switching away, remove geo overlays from main map
                    // but keep them stored for re-display
                }
            });
        });
    }

    showLocalMap() {
        const mapEl = document.getElementById('map');
        if (!mapEl) return;

        // Hide the main Leaflet map
        if (window.mapManager?.map) {
            window.mapManager.map.getContainer().style.display = 'none';
        }

        // Create local map container if it doesn't exist
        let localMapEl = document.getElementById('local-map');
        if (!localMapEl) {
            localMapEl = document.createElement('div');
            localMapEl.id = 'local-map';
            localMapEl.style.cssText = 'width:100%;height:100%;position:absolute;top:0;left:0;background:#1a1a2e;';
            mapEl.parentElement.appendChild(localMapEl);
        }
        localMapEl.style.display = 'block';

        if (!this.localMap) {
            this.localMap = L.map('local-map', {
                crs: L.CRS.Simple,
                minZoom: -5,
                maxZoom: 5,
                zoomSnap: 0.25,
                attributionControl: false,
            });
            this.localMap.setView([0, 0], 0);
        }

        this.localMapActive = true;
        this.localMap.invalidateSize();

        // Re-display current overlay if any
        if (this.currentOverlay) {
            if (!this.localMap.hasLayer(this.currentOverlay)) {
                this.currentOverlay.addTo(this.localMap);
            }
        }

        // Apply any active symbology rendering filter (gamma/contrast/brightness/saturation)
        // to the freshly created panes. The controller may have been adjusted before the
        // user actually loaded a raster.
        if (window.symbology && window.symbology.local) {
            window.symbology.local._applyRenderingToLocalMap();
        }
    }

    hideLocalMap() {
        const localMapEl = document.getElementById('local-map');
        if (localMapEl) {
            localMapEl.style.display = 'none';
        }

        // Show the main map again
        if (window.mapManager?.map) {
            window.mapManager.map.getContainer().style.display = '';
            window.mapManager.map.invalidateSize();
        }

        this.localMapActive = false;
    }

    // ===== Data loading (Band Registration mode) =====

    async loadImages() {
        try {
            const res = await fetch('/api/local/images');
            const data = await res.json();
            const select = document.getElementById('local-image-select');
            if (!select) return;
            select.innerHTML = '<option value="">-- Select Image --</option>';
            data.images.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name.replace(/_/g, ' ');
                select.appendChild(opt);
            });
        } catch (e) {
            console.error('Failed to load local images:', e);
        }
    }

    setupSelectors() {
        const imageSelect = document.getElementById('local-image-select');
        const algoSelect = document.getElementById('local-algo-select');

        if (imageSelect) {
            imageSelect.addEventListener('change', async () => {
                this.currentImageDir = imageSelect.value;
                algoSelect.disabled = !this.currentImageDir;
                algoSelect.innerHTML = '<option value="">-- Select Algorithm --</option>';
                document.getElementById('local-band-controls').style.display = 'none';
                this._bandRegBandsLoaded = false;
                this.clearOverlay();

                if (!this.currentImageDir) return;

                try {
                    const res = await fetch(`/api/local/algorithms/${encodeURIComponent(this.currentImageDir)}`);
                    const data = await res.json();
                    data.algorithms.forEach(name => {
                        const opt = document.createElement('option');
                        opt.value = name;
                        opt.textContent = name;
                        algoSelect.appendChild(opt);
                    });
                } catch (e) {
                    console.error('Failed to load algorithms:', e);
                }
            });
        }

        if (algoSelect) {
            algoSelect.addEventListener('change', async () => {
                this.currentAlgoDir = algoSelect.value;
                if (!this.currentAlgoDir) {
                    document.getElementById('local-band-controls').style.display = 'none';
                    this._bandRegBandsLoaded = false;
                    return;
                }
                await this.loadBands();
            });
        }
    }

    async loadBands() {
        this.platform.showLoading('Loading bands...');
        try {
            // Clear previous state when switching algorithm
            this.clearOverlay();
            this.preloadedBandUrls = {};
            this.preloadedSize = { width: 0, height: 0 };
            this._bandImages = {};

            const res = await fetch(`/api/local/bands/${encodeURIComponent(this.currentImageDir)}/${encodeURIComponent(this.currentAlgoDir)}`);
            const data = await res.json();
            this.availableBands = data.bands;

            // Populate band pool
            const pool = document.getElementById('local-band-pool');
            pool.innerHTML = '';
            this.availableBands.forEach(band => {
                const chip = document.createElement('div');
                chip.className = 'band-chip';
                chip.draggable = true;
                chip.dataset.band = band;
                chip.textContent = band.replace('after_', '').replace('.tif', '');
                chip.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', band);
                    chip.classList.add('dragging');
                });
                chip.addEventListener('dragend', () => chip.classList.remove('dragging'));
                pool.appendChild(chip);
            });

            // Set default RGB slots
            const defaults = this.availableBands.slice(0, 3);
            ['local-slot-r', 'local-slot-g', 'local-slot-b'].forEach((slotId, i) => {
                const slotBand = document.querySelector(`#${slotId} .slot-band`);
                if (slotBand && defaults[i]) {
                    slotBand.textContent = defaults[i].replace('after_', '').replace('.tif', '');
                    slotBand.dataset.band = defaults[i];
                }
            });

            document.getElementById('local-band-controls').style.display = '';
            this._bandRegBandsLoaded = true;
            this.setupShortcutSlots();

            // GPU: load to GPU + compute percentiles + stretch in one call
            await this.gpuLoadBands();
        } catch (e) {
            console.error('Failed to load bands:', e);
            this.platform.hideLoading();
        }
    }

    async loadPercentiles() {
        try {
            const res = await fetch('/api/local/percentiles', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_dir: this.currentImageDir,
                    algorithm_dir: this.currentAlgoDir
                })
            });
            if (res.ok) {
                const data = await res.json();
                this.percentiles = data.percentiles;
                this._updateMinMaxFromSliders();
            }
        } catch (e) {
            console.error('Failed to load percentiles:', e);
        }
    }

    setupPercentileSlider() {
        const lowSlider = document.getElementById('local-pct-low');
        const highSlider = document.getElementById('local-pct-high');
        const lowInput = document.getElementById('local-pct-low-val');
        const highInput = document.getElementById('local-pct-high-val');
        if (!lowSlider || !highSlider) return;

        const sync = (source) => this._updateMinMaxFromSliders(source);
        lowSlider.addEventListener('input', () => sync('slider'));
        highSlider.addEventListener('input', () => sync('slider'));
        lowInput.addEventListener('change', () => sync('input'));
        highInput.addEventListener('change', () => sync('input'));
    }

    _updateMinMaxFromSliders(source) {
        const lowSlider = document.getElementById('local-pct-low');
        const highSlider = document.getElementById('local-pct-high');
        const lowInput = document.getElementById('local-pct-low-val');
        const highInput = document.getElementById('local-pct-high-val');
        const minHidden = document.getElementById('local-min');
        const maxHidden = document.getElementById('local-max');
        if (!lowSlider || !highSlider) return;

        let lo, hi;
        if (source === 'slider') {
            lo = parseFloat(lowSlider.value);
            hi = parseFloat(highSlider.value);
            if (lo > hi) { lowSlider.value = hi; lo = hi; }
            if (hi < lo) { highSlider.value = lo; hi = lo; }
            lowInput.value = lo.toFixed(1);
            highInput.value = hi.toFixed(1);
        } else {
            lo = Math.max(0, Math.min(100, parseFloat(lowInput.value) || 0));
            hi = Math.max(0, Math.min(100, parseFloat(highInput.value) || 100));
            if (lo > hi) lo = hi;
            lowSlider.value = lo;
            highSlider.value = hi;
        }

        if (this.percentiles) {
            const loKey = String(Math.max(1, Math.round(lo)));
            const hiKey = String(Math.min(99, Math.round(hi)));
            minHidden.value = Math.round(this.percentiles[loKey] || 0);
            maxHidden.value = Math.round(this.percentiles[hiKey] || 3000);
        }
    }

    async preloadAllBands() {
        const min = parseFloat(document.getElementById('local-min')?.value) || 0;
        const max = parseFloat(document.getElementById('local-max')?.value) || 3000;
        this._preloadedMin = min;
        this._preloadedMax = max;

        try {
            const res = await fetch('/api/local/preload-bands', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_dir: this.currentImageDir,
                    algorithm_dir: this.currentAlgoDir,
                    min_val: min,
                    max_val: max
                })
            });
            if (res.ok) {
                const data = await res.json();
                this.preloadedBandUrls = data.band_urls;
                this.preloadedSize = { width: data.width, height: data.height };

                this._bandImages = {};
                this._bandOverlays = {};
                if (!this.localMap) this.showLocalMap();
                const bounds = [[0, 0], [data.height, data.width]];

                const loadPromises = Object.entries(data.band_urls).map(([band, url]) => {
                    return new Promise((resolve) => {
                        const img = new window.Image();
                        img.crossOrigin = 'anonymous';
                        img.onload = () => {
                            this._bandImages[band] = img;
                            const overlay = L.imageOverlay(url, bounds, { opacity: 1.0, interactive: false });
                            overlay.addTo(this.localMap);
                            overlay.getElement().style.display = 'none';
                            this._bandOverlays[band] = overlay;
                            resolve();
                        };
                        img.onerror = () => resolve();
                        img.src = url;
                    });
                });
                await Promise.all(loadPromises);

                this._autoShowDefaultRGB();
                this.platform.hideLoading();
                this.platform.showNotification('Bands ready', 'success', 1500);
            }
        } catch (e) {
            console.error('Preload failed:', e);
            this.platform.hideLoading();
        }
    }

    // =================================================================
    // GPU-accelerated band loading and stretch (Band Registration)
    // =================================================================

    async gpuLoadBands() {
        const min = parseFloat(document.getElementById('local-min')?.value) || 0;
        const max = parseFloat(document.getElementById('local-max')?.value) || 3000;
        this._preloadedMin = min;
        this._preloadedMax = max;

        try {
            const res = await fetch('/api/local/gpu-load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_dir: this.currentImageDir,
                    algorithm_dir: this.currentAlgoDir,
                    min_val: min,
                    max_val: max
                })
            });
            if (!res.ok) {
                console.error('gpu-load failed, falling back to legacy');
                await this.loadPercentiles();
                await this.preloadAllBands();
                return;
            }

            const data = await res.json();
            this.percentiles = data.percentiles;
            this._updateMinMaxFromSliders();
            this.preloadedBandUrls = data.band_urls;
            this.preloadedSize = { width: data.width, height: data.height };

            await this._loadBandImages(data.band_urls, data.width, data.height);
            this._autoShowDefaultRGB();
            this.platform.hideLoading();
            this.platform.showNotification('Bands ready (GPU)', 'success', 1500);
        } catch (e) {
            console.error('gpuLoadBands failed:', e);
            await this.loadPercentiles();
            await this.preloadAllBands();
        }
    }

    async gpuStretchBands() {
        const min = parseFloat(document.getElementById('local-min')?.value) || 0;
        const max = parseFloat(document.getElementById('local-max')?.value) || 3000;
        this._preloadedMin = min;
        this._preloadedMax = max;

        try {
            const res = await fetch('/api/local/gpu-stretch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_dir: this.currentImageDir,
                    algorithm_dir: this.currentAlgoDir,
                    min_val: min,
                    max_val: max
                })
            });
            if (!res.ok) {
                console.warn('gpu-stretch failed, falling back to full preload');
                await this.preloadAllBands();
                return;
            }

            const data = await res.json();
            this.preloadedBandUrls = data.band_urls;
            this.preloadedSize = { width: data.width, height: data.height };
            await this._loadBandImages(data.band_urls, data.width, data.height);
        } catch (e) {
            console.error('gpuStretchBands failed:', e);
            await this.preloadAllBands();
        }
    }

    async _loadBandImages(bandUrls, width, height) {
        this._clearBandOverlays();
        this._bandImages = {};
        this._bandOverlays = {};
        if (!this.localMap) this.showLocalMap();
        const bounds = [[0, 0], [height, width]];

        const loadPromises = Object.entries(bandUrls).map(([band, url]) => {
            return new Promise((resolve) => {
                const img = new window.Image();
                img.crossOrigin = 'anonymous';
                img.onload = () => {
                    this._bandImages[band] = img;
                    const overlay = L.imageOverlay(url, bounds, { opacity: 1.0, interactive: false });
                    overlay.addTo(this.localMap);
                    overlay.getElement().style.display = 'none';
                    this._bandOverlays[band] = overlay;
                    resolve();
                };
                img.onerror = () => resolve();
                img.src = url;
            });
        });
        await Promise.all(loadPromises);
    }

    _autoShowDefaultRGB() {
        const rBand = document.querySelector('#local-slot-r .slot-band')?.dataset.band;
        const gBand = document.querySelector('#local-slot-g .slot-band')?.dataset.band;
        const bBand = document.querySelector('#local-slot-b .slot-band')?.dataset.band;
        if (!rBand || !gBand || !bBand) return;

        const dataUrl = this._composeRGBFromBands(rBand, gBand, bBand);
        if (dataUrl && this.preloadedSize.width) {
            this.displayOverlay(dataUrl, this.preloadedSize.width, this.preloadedSize.height);
        }
    }

    _clearBandOverlays() {
        if (this._bandOverlays) {
            Object.values(this._bandOverlays).forEach(ov => {
                if (this.localMap?.hasLayer(ov)) this.localMap.removeLayer(ov);
            });
        }
        this._bandOverlays = {};
    }

    /**
     * Read per-channel (min, max) from the symbology controller. Falls back to the
     * legacy hidden inputs (single value applied to all channels) when the symbology
     * UI is not in 'minmax' mode or its state is not yet initialized.
     */
    _readSymbologyRanges() {
        const sc = window.symbology && window.symbology.local;
        if (sc && sc.state && sc.state.bands && sc.state.mode === 'minmax') {
            return {
                r: { min: Number(sc.state.bands.r.min), max: Number(sc.state.bands.r.max) },
                g: { min: Number(sc.state.bands.g.min), max: Number(sc.state.bands.g.max) },
                b: { min: Number(sc.state.bands.b.min), max: Number(sc.state.bands.b.max) },
            };
        }
        const fallbackMin = parseFloat(document.getElementById('local-min')?.value) || 0;
        const fallbackMax = parseFloat(document.getElementById('local-max')?.value) || 3000;
        return {
            r: { min: fallbackMin, max: fallbackMax },
            g: { min: fallbackMin, max: fallbackMax },
            b: { min: fallbackMin, max: fallbackMax },
        };
    }

    /**
     * Per-band stretch + composite for upload mode. Refetches only the channels whose
     * (band, min, max) tuple changed since the last call, then RGB-composites the
     * three slot images on a canvas.
     */
    async _composePerBandUpload(rBand, gBand, bBand, ranges, hooks) {
        if (!this.uploadId) return null;
        const showLoad = hooks?.showLoad || (() => {});
        const hideLoad = hooks?.hideLoad || (() => {});

        const wanted = {
            r: { band: rBand, min: ranges.r.min, max: ranges.r.max },
            g: { band: gBand, min: ranges.g.min, max: ranges.g.max },
            b: { band: bBand, min: ranges.b.min, max: ranges.b.max },
        };

        // Determine which slots need re-stretching.
        const stale = [];
        for (const slot of ['r', 'g', 'b']) {
            const cur = this._slotMeta[slot];
            const w = wanted[slot];
            if (!cur || cur.band !== w.band || cur.min !== w.min || cur.max !== w.max) {
                stale.push(slot);
            }
        }

        if (stale.length > 0) {
            showLoad(`Stretching ${stale.length === 3 ? 'channels' : stale.join('/').toUpperCase()}…`);
            try {
                const specs = stale.map(slot => ({
                    slot,
                    band_name: wanted[slot].band,
                    min_val: wanted[slot].min,
                    max_val: wanted[slot].max,
                }));
                const res = await fetch('/api/local/uploaded/gpu-stretch-multi', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ upload_id: this.uploadId, specs }),
                });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                this.preloadedSize = { width: data.width, height: data.height };

                await Promise.all(stale.map(slot => new Promise(resolve => {
                    const url = data.slot_urls?.[slot];
                    if (!url) return resolve();
                    const img = new window.Image();
                    img.crossOrigin = 'anonymous';
                    img.onload = () => {
                        this._slotImages[slot] = img;
                        this._slotMeta[slot] = { ...wanted[slot] };
                        resolve();
                    };
                    img.onerror = () => resolve();
                    img.src = url;
                })));
            } catch (err) {
                console.error('gpu-stretch-multi failed:', err);
                hideLoad();
                return null;
            }
            hideLoad();
        }

        const rImg = this._slotImages.r;
        const gImg = this._slotImages.g;
        const bImg = this._slotImages.b;
        if (!rImg || !gImg || !bImg) return null;

        const w = rImg.naturalWidth;
        const h = rImg.naturalHeight;
        if (!w || !h) return null;

        const offscreen = document.createElement('canvas');
        offscreen.width = w;
        offscreen.height = h;
        const ctx = offscreen.getContext('2d');

        ctx.drawImage(rImg, 0, 0);
        const rData = ctx.getImageData(0, 0, w, h).data;
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(gImg, 0, 0);
        const gData = ctx.getImageData(0, 0, w, h).data;
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(bImg, 0, 0);
        const bData = ctx.getImageData(0, 0, w, h).data;

        const result = ctx.createImageData(w, h);
        const out = result.data;
        for (let i = 0; i < rData.length; i += 4) {
            out[i]     = rData[i];
            out[i + 1] = gData[i];
            out[i + 2] = bData[i];
            out[i + 3] = 255;
        }
        ctx.putImageData(result, 0, 0);
        const dataUrl = offscreen.toDataURL('image/png');
        this.displayOverlay(dataUrl, w, h);
        return dataUrl;
    }

    _composeRGBFromBands(rBand, gBand, bBand) {
        const rImg = this._bandImages?.[rBand];
        const gImg = this._bandImages?.[gBand];
        const bImg = this._bandImages?.[bBand];
        if (!rImg || !gImg || !bImg) return null;

        const w = rImg.naturalWidth;
        const h = rImg.naturalHeight;
        if (!w || !h) return null;

        const offscreen = document.createElement('canvas');
        offscreen.width = w;
        offscreen.height = h;
        const ctx = offscreen.getContext('2d');

        ctx.drawImage(rImg, 0, 0);
        const rData = ctx.getImageData(0, 0, w, h).data;
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(gImg, 0, 0);
        const gData = ctx.getImageData(0, 0, w, h).data;
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(bImg, 0, 0);
        const bData = ctx.getImageData(0, 0, w, h).data;

        const result = ctx.createImageData(w, h);
        const out = result.data;
        for (let i = 0; i < rData.length; i += 4) {
            out[i]     = rData[i];
            out[i + 1] = gData[i];
            out[i + 2] = bData[i];
            out[i + 3] = 255;
        }
        ctx.putImageData(result, 0, 0);
        return offscreen.toDataURL('image/png');
    }

    // ===== Drag & Drop =====

    setupDragDrop() {
        const slots = document.querySelectorAll('#local-slot-r, #local-slot-g, #local-slot-b');
        slots.forEach(slot => {
            slot.addEventListener('dragover', (e) => { e.preventDefault(); slot.classList.add('drag-over'); });
            slot.addEventListener('dragleave', () => slot.classList.remove('drag-over'));
            slot.addEventListener('drop', (e) => {
                e.preventDefault();
                slot.classList.remove('drag-over');
                const band = e.dataTransfer.getData('text/plain');
                const slotBand = slot.querySelector('.slot-band');
                if (slotBand && band) {
                    // Shorten display: "Band 1" → "1", "after_xx.tif" → "xx"
                    if (this.currentMode === 'load-image') {
                        slotBand.textContent = band.replace(/^Band\s*/i, '');
                    } else {
                        slotBand.textContent = band.replace('after_', '').replace('.tif', '');
                    }
                    slotBand.dataset.band = band;
                }
            });
        });
    }

    // ===== Visualization =====

    setupApplyButton() {
        const btn = document.getElementById('apply-local-vis');
        if (btn) {
            btn.addEventListener('click', () => this.applyRGBVisualization({ silent: true }));
        }
        // Visualization changes (Min/Max, mode, percentile) intentionally do NOT auto-apply.
        // The user must click Apply, matching the cloud (S2/S1) behavior. Rendering changes
        // (gamma/contrast/brightness/saturation) remain instant via the symbology controller's
        // CSS-filter pipeline — see `_applyRenderingToLocalOverlays` in symbology-controller.js.
    }

    async applyRGBVisualization(opts) {
        const silent = !!(opts && opts.silent);
        const notify = (msg, level, ms) => { if (!silent) this.platform.showNotification(msg, level, ms); };
        const showLoad = (msg) => { if (!silent) this.platform.showLoading(msg); };
        const hideLoad = () => { if (!silent) this.platform.hideLoading(); };

        if (this.currentMode === 'load-image') {
            if (!this.uploadId) {
                notify('Upload a GeoTIFF first', 'warning');
                return;
            }
        } else {
            if (!this.currentImageDir || !this.currentAlgoDir) {
                notify('Select image and algorithm first', 'warning');
                return;
            }
        }

        const rBand = document.querySelector('#local-slot-r .slot-band')?.dataset.band;
        const gBand = document.querySelector('#local-slot-g .slot-band')?.dataset.band;
        const bBand = document.querySelector('#local-slot-b .slot-band')?.dataset.band;

        if (!rBand || !gBand || !bBand) {
            notify('Assign bands to R, G, B slots', 'warning');
            return;
        }

        // Load-image mode supports true per-band Min/Max via gpu-stretch-multi.
        // Each channel can have its own (band, min, max) tuple — falls back to the
        // single-range path below for band-registration mode.
        if (this.currentMode === 'load-image') {
            const ranges = this._readSymbologyRanges();
            const dataUrl = await this._composePerBandUpload(rBand, gBand, bBand, ranges, { showLoad, hideLoad });
            if (dataUrl) {
                notify(`RGB: ${rBand} / ${gBand} / ${bBand}`, 'success', 2000);
            }
            return;
        }

        // Band-registration: existing single-stretch flow.
        const curMin = parseFloat(document.getElementById('local-min')?.value) || 0;
        const curMax = parseFloat(document.getElementById('local-max')?.value) || 3000;
        if (this._preloadedMin !== curMin || this._preloadedMax !== curMax) {
            showLoad('Applying new stretch (GPU)...');
            await this.gpuStretchBands();
            hideLoad();
        }

        // Try instant client-side compositing
        const dataUrl = this._composeRGBFromBands(rBand, gBand, bBand);
        if (dataUrl && this.preloadedSize.width) {
            this.displayOverlay(dataUrl, this.preloadedSize.width, this.preloadedSize.height);
            const label = `RGB: ${rBand.replace('after_','').replace('.tif','')} / ${gBand.replace('after_','').replace('.tif','')} / ${bBand.replace('after_','').replace('.tif','')}`;
            notify(label, 'success', 2000);
            return;
        }

        // Fallback: server-side (band registration only)
        if (this.currentMode === 'band-registration') {
            const min = parseFloat(document.getElementById('local-min')?.value) || 0;
            const max = parseFloat(document.getElementById('local-max')?.value) || 3000;
            showLoad('Creating visualization...');
            try {
                const res = await fetch('/api/local/visualize-rgb', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_dir: this.currentImageDir,
                        algorithm_dir: this.currentAlgoDir,
                        bands: [rBand, gBand, bBand],
                        min_val: min,
                        max_val: max
                    })
                });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                this.displayOverlay(data.image_url, data.width, data.height);
            } catch (e) {
                console.error('RGB visualization error:', e);
                notify('Visualization failed', 'error');
            } finally {
                hideLoad();
            }
        }
    }

    applyGrayscaleBand(bandFile) {
        if (this.currentMode === 'load-image') {
            if (!this.uploadId) return;
            // Use geo band overlay on main map
            const overlay = this._geoBandOverlays?.[bandFile];
            if (overlay) {
                this._hideCurrentGeoOverlay();
                overlay.getElement().style.display = '';
                overlay._isBandOverlay = true;
                this.geoOverlay = overlay;
                return;
            }
        } else {
            if (!this.currentImageDir || !this.currentAlgoDir) return;
            // Use pre-created Leaflet overlay on CRS.Simple map
            const overlay = this._bandOverlays?.[bandFile];
            if (overlay) {
                if (this.currentOverlay && this.currentOverlay !== overlay) {
                    if (this.currentOverlay._isBandOverlay) {
                        this.currentOverlay.getElement().style.display = 'none';
                    } else {
                        if (this.localMap?.hasLayer(this.currentOverlay)) {
                            this.localMap.removeLayer(this.currentOverlay);
                        }
                    }
                }
                overlay.getElement().style.display = '';
                overlay._isBandOverlay = true;
                this.currentOverlay = overlay;

                if (!this._hasInitialFit) {
                    this.localMap.fitBounds([[0, 0], [this.preloadedSize.height, this.preloadedSize.width]]);
                    this._hasInitialFit = true;
                }
                return;
            }

            // Fallback: use preloaded URL
            const preloadedUrl = this.preloadedBandUrls?.[bandFile];
            if (preloadedUrl && this.preloadedSize.width) {
                this.displayOverlay(preloadedUrl, this.preloadedSize.width, this.preloadedSize.height);
            }
        }
    }

    _hideCurrentGeoOverlay() {
        const map = window.mapManager?.map;
        if (!map) return;
        if (this.geoOverlay) {
            if (this.geoOverlay._isBandOverlay) {
                this.geoOverlay.getElement().style.display = 'none';
            } else if (map.hasLayer(this.geoOverlay)) {
                map.removeLayer(this.geoOverlay);
            }
        }
    }

    clearOverlay() {
        if (this.currentMode === 'load-image') {
            this._hideCurrentGeoOverlay();
            this.geoOverlay = null;
            this._dispatchLayerEvent('layer:removed', { id: 'local-base-image', type: 'image' });
            // Clear geo band overlays
            if (this._geoBandOverlays) {
                const map = window.mapManager?.map;
                Object.values(this._geoBandOverlays).forEach(ov => {
                    if (map?.hasLayer(ov)) map.removeLayer(ov);
                });
                this._geoBandOverlays = {};
            }
        } else {
            if (this.currentOverlay && this.localMap) {
                if (!this.currentOverlay._isBandOverlay) {
                    this.localMap.removeLayer(this.currentOverlay);
                }
                this.currentOverlay = null;
            }
            this._clearBandOverlays();
        }
        this._hasInitialFit = false;
    }

    displayOverlay(dataUrl, width, height) {
        if (this.currentMode === 'load-image') {
            // Display on main map with geo bounds
            const map = window.mapManager?.map;
            if (!map || !this.geoBounds) return;

            this._showMainMap();
            this._hideCurrentGeoOverlay();

            const newOverlay = L.imageOverlay(dataUrl, this.geoBounds, {
                opacity: 1.0,
                interactive: false,
            });
            newOverlay.addTo(map);
            this.geoOverlay = newOverlay;
            this._dispatchLayerEvent('layer:added', { id: 'local-base-image', type: 'image', name: 'Local Image', layer: newOverlay });

            if (!this._hasInitialFit) {
                map.fitBounds(this.geoBounds);
                this._hasInitialFit = true;
            }
        } else {
            // Display on CRS.Simple local map
            if (!this.localMap) this.showLocalMap();

            const isFirstLoad = !this._hasInitialFit;
            const oldOverlay = this.currentOverlay;
            const bounds = [[0, 0], [height, width]];

            if (oldOverlay) {
                if (oldOverlay._isBandOverlay) {
                    oldOverlay.getElement().style.display = 'none';
                } else if (this.localMap.hasLayer(oldOverlay)) {
                    this.localMap.removeLayer(oldOverlay);
                }
            }

            const newOverlay = L.imageOverlay(dataUrl, bounds, {
                opacity: 1.0,
                interactive: false,
            });
            newOverlay.addTo(this.localMap);
            this.currentOverlay = newOverlay;

            if (isFirstLoad) {
                this.localMap.fitBounds(bounds);
                this._hasInitialFit = true;
            }
        }
    }

    // ===== Keyboard Shortcut Mode =====

    setupShortcutSlots() {
        const container = document.getElementById('local-keyboard-shortcut-slots');
        if (!container) return;
        container.innerHTML = '';
        this.keyboardShortcutMap = {};

        this.availableBands.forEach((band, i) => {
            if (i >= 9) return;
            const key = String(i + 1);
            this.keyboardShortcutMap[key] = band;

            const displayName = this.currentMode === 'load-image'
                ? band
                : band.replace('after_', '').replace('.tif', '');

            const slot = document.createElement('div');
            slot.className = 'shortcut-slot';
            slot.dataset.key = key;
            slot.dataset.band = band;
            slot.innerHTML = `<span class="slot-key">${key}</span><span class="slot-band-label" data-band="${band}">${displayName}</span>`;

            slot.addEventListener('dragover', (e) => { e.preventDefault(); slot.classList.add('drag-over'); });
            slot.addEventListener('dragleave', () => slot.classList.remove('drag-over'));
            slot.addEventListener('drop', (e) => {
                e.preventDefault();
                slot.classList.remove('drag-over');
                const droppedBand = e.dataTransfer.getData('text/plain');
                if (!droppedBand) return;
                slot.dataset.band = droppedBand;
                const dn = this.currentMode === 'load-image'
                    ? droppedBand
                    : droppedBand.replace('after_', '').replace('.tif', '');
                slot.querySelector('.slot-band-label').textContent = dn;
                slot.querySelector('.slot-band-label').dataset.band = droppedBand;
                this.keyboardShortcutMap[key] = droppedBand;
            });

            container.appendChild(slot);
        });
    }

    setupKeyboardShortcut() {
        const toggle = document.getElementById('local-keyboard-shortcut-toggle');
        const container = document.getElementById('local-keyboard-shortcut-slots');
        if (!toggle || !container) return;

        toggle.addEventListener('change', () => {
            this.keyboardShortcutMode = toggle.checked;
            container.style.display = toggle.checked ? 'grid' : 'none';
        });

        document.addEventListener('keydown', (e) => {
            if (!this.keyboardShortcutMode) return;
            const tabContent = document.getElementById('local-image-tab-content');
            if (!tabContent || !tabContent.classList.contains('active')) return;

            const tag = document.activeElement?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            if (document.activeElement?.isContentEditable) return;

            if (e.key >= '1' && e.key <= '9') {
                const band = this.keyboardShortcutMap[e.key];
                if (!band) return;
                e.preventDefault();

                const slot = container.querySelector(`.shortcut-slot[data-key="${e.key}"]`);
                if (slot) {
                    slot.classList.add('active-key');
                    setTimeout(() => slot.classList.remove('active-key'), 300);
                }

                const label = this.currentMode === 'load-image'
                    ? band
                    : band.replace('after_', '').replace('.tif', '');
                this.platform.showNotification(`Band ${label} (key ${e.key})`, 'info', 2000);
                this.applyGrayscaleBand(band);
            }
            if (e.key === '`' || e.key === '~') {
                e.preventDefault();
                this._autoShowDefaultRGB();
                this.platform.showNotification('Restored RGB (~)', 'info', 2000);
            }
        });
    }

    // ===== Local Analysis =====

    async runLocalAnalysis() {
        if (this.currentMode === 'band-registration') {
            this.platform.showNotification('Analysis not available in Band Registration mode', 'warning');
            return;
        }

        if (this.currentMode === 'load-image') {
            if (!this.uploadId) {
                this.platform.showNotification('Upload a GeoTIFF first', 'warning');
                return;
            }
        } else {
            if (!this.currentImageDir || !this.currentAlgoDir) {
                this.platform.showNotification('Select image and algorithm first', 'warning');
                return;
            }
        }

        this.lastLocalAnalysisResults = {};
        this.showLocalAnalysisResults(this.lastLocalAnalysisResults);
        this.platform.showNotification('Analysis panel ready', 'success');

        if (this.platform.uiManager) {
            this.platform.uiManager.switchToAnalysisTab();
        } else {
            this.platform.switchToAnalysisTab();
        }
    }

    showLocalAnalysisResults(analysisResults) {
        this.lastLocalAnalysisResults = analysisResults;
        // User-triggered flow: clear any existing overlays, then rebuild.
        this.clearAllAnalysisOverlays();
        this._renderLocalAnalysisListDOM();
    }

    /**
     * Rebuild the `.analysis-list` DOM from `this.lastLocalAnalysisResults`
     * WITHOUT clearing existing overlays. Used by `handleSlotChange` so a
     * slot swap doesn't nuke the overlays we just restored from stash.
     */
    _renderLocalAnalysisListDOM() {
        const analysisList = document.querySelector('.analysis-list');
        if (!analysisList) return;

        analysisList.innerHTML = '';
        this.addLocalSpectralAnalysisOption(analysisList);
        this.addLocalTargetDetectionOption(analysisList);
        if (this.isUploadedImageActive) {
            this.addLocalSAM3Option(analysisList);
        }
    }

    addLocalSpectralAnalysisOption(analysisList) {
        const saItem = document.createElement('div');
        saItem.className = 'analysis-item spectral-analysis-option';
        saItem.dataset.modelId = 'spectral-analysis';
        saItem.dataset.active = 'false';

        saItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);">
                <div class="custom-icon">📊</div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">Spectral Analysis</h4>
                <div class="analysis-status inactive">Click to start</div>
            </div>
        `;

        saItem.onclick = (e) => {
            if (e.target.closest('.sa-ui')) return;
            if (!this.platform.spectralAnalysis) {
                this.platform.spectralAnalysis = new SpectralAnalysisController(this.platform);
            }
            this.platform.spectralAnalysis.handleItemClick(saItem);
        };

        analysisList.appendChild(saItem);
    }

    addLocalCustomVizOption(analysisList) {
        const customItem = document.createElement('div');
        customItem.className = 'analysis-item custom-visualization';
        customItem.dataset.modelId = 'custom';
        customItem.dataset.active = 'false';
        customItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder">
                <div class="custom-icon">🛠️</div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">Custom Visualization</h4>
                <p class="analysis-subtitle">Create custom visualizations</p>
                <div class="analysis-status inactive">Click to create</div>
            </div>
        `;
        customItem.onclick = () => {
            if (this.platform.customViz) {
                this.platform.customViz.showLocalModal(this.currentImageDir, this.currentAlgoDir, this.availableBands);
            }
        };
        analysisList.appendChild(customItem);
    }

    addLocalTargetDetectionOption(analysisList) {
        const tdItem = document.createElement('div');
        tdItem.className = 'analysis-item target-detection-option';
        tdItem.dataset.modelId = 'target-detection';
        tdItem.dataset.active = 'false';
        tdItem.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);">
                <div class="custom-icon">🎯</div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">Target Detection</h4>
                <div class="analysis-status inactive">Click to start</div>
            </div>
        `;
        tdItem.onclick = (e) => {
            if (e.target.closest('.td-ui')) return;
            if (!this.platform.targetDetection) {
                this.platform.showNotification('Target Detection module not loaded', 'error');
                return;
            }
            this.platform.targetDetection.handleItemClick();
        };
        analysisList.appendChild(tdItem);
    }

    addLocalSAM3Option(analysisList) {
        const sam3Item = document.createElement('div');
        sam3Item.className = 'analysis-item sam3-option';
        sam3Item.dataset.modelId = 'sam3';
        sam3Item.dataset.active = 'false';
        sam3Item.innerHTML = `
            <div class="analysis-thumbnail custom-placeholder" style="background: linear-gradient(135deg, #7b1fa2 0%, #e91e63 100%);">
                <div class="custom-icon">
                    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="white" stroke-width="2">
                        <path d="M6 9l6 6 6-6"/>
                        <rect x="3" y="3" width="18" height="18" rx="3"/>
                        <circle cx="8" cy="8" r="1.5" fill="white"/>
                        <circle cx="16" cy="16" r="1.5" fill="white"/>
                    </svg>
                </div>
            </div>
            <div class="analysis-info">
                <h4 class="analysis-title">SAM3 Segmentation</h4>
                <div class="analysis-status inactive">Click to start</div>
            </div>
        `;
        sam3Item.onclick = (e) => {
            if (e.target.closest('.sam3-ui')) return;
            if (!this.platform.sam3Controller) {
                this.platform.showNotification('SAM3 module not loaded', 'error');
                return;
            }
            this.platform.sam3Controller.handleItemClick();
        };
        analysisList.appendChild(sam3Item);
    }

    // ===== Analysis Overlay Management (mode-aware) =====

    showLocalAnalysisLayer(modelId, imageUrl, imgW, imgH, displayName, isBinary = false) {
        this.hideLocalAnalysisLayer(modelId);

        if (this.currentMode === 'load-image') {
            // load-image mode: delegate to mapManager for unified behavior
            // (same opacity, interactive, event dispatch as satellite search)
            if (!window.mapManager || !this.geoBounds) return;
            window.mapManager.showAnalysisLayer(modelId, imageUrl, displayName || modelId, this.geoBounds, isBinary);
            // Keep a reference so we can clean up later
            this.geoAnalysisOverlays[modelId] = window.mapManager.analysisLayers?.[modelId] || null;
        } else {
            // band-registration mode: CRS.Simple local map (no change)
            if (!this.localMap) this.showLocalMap();
            const bounds = [[0, 0], [imgH || this.preloadedSize.height, imgW || this.preloadedSize.width]];
            const layer = L.imageOverlay(imageUrl, bounds, { opacity: 1.0, interactive: false });
            layer.addTo(this.localMap);
            this.localAnalysisOverlays[modelId] = layer;
        }
    }

    hideLocalAnalysisLayer(modelId) {
        if (this.currentMode === 'load-image') {
            // Delegate to mapManager (handles event dispatch, layer cleanup)
            if (window.mapManager) {
                window.mapManager.hideAnalysisLayer(modelId);
            }
            delete this.geoAnalysisOverlays[modelId];
        } else {
            if (this.localAnalysisOverlays[modelId] && this.localMap) {
                this.localMap.removeLayer(this.localAnalysisOverlays[modelId]);
                delete this.localAnalysisOverlays[modelId];
            }
        }
    }

    clearAllAnalysisOverlays() {
        // Clear band-registration local map overlays
        Object.keys(this.localAnalysisOverlays).forEach(id => {
            if (this.localAnalysisOverlays[id] && this.localMap) {
                this.localMap.removeLayer(this.localAnalysisOverlays[id]);
            }
        });
        this.localAnalysisOverlays = {};

        // Clear load-image overlays via mapManager
        Object.keys(this.geoAnalysisOverlays).forEach(id => {
            if (window.mapManager) {
                window.mapManager.hideAnalysisLayer(id);
            }
        });
        this.geoAnalysisOverlays = {};
    }

    // ===== Local Colorbar Threshold =====

    setupLocalColorbarThreshold(analysisItem, modelId, result, imgW, imgH) {
        const container = analysisItem.querySelector('.colorbar-container');
        if (!container) return;

        const track = container.querySelector('.colorbar-track');
        const minHandle = container.querySelector('.min-handle');
        const maxHandle = container.querySelector('.max-handle');
        const selection = container.querySelector('.colorbar-selection');
        const minInput = container.querySelector('.colorbar-min-input');
        const maxInput = container.querySelector('.colorbar-max-input');
        const applyBtn = container.querySelector('.colorbar-apply-btn');
        const cancelBtn = container.querySelector('.colorbar-cancel-btn');

        if (!track || !minHandle || !maxHandle) return;

        const colormap = result.colormap || { min_val: -1, max_val: 1 };
        const minVal = colormap.min_val;
        const maxVal = colormap.max_val;
        let currentMin = minVal;
        let currentMax = maxVal;

        const updateUI = () => {
            const trackWidth = track.offsetWidth;
            const range = maxVal - minVal;
            if (range === 0 || trackWidth === 0) return;
            const minPos = ((currentMin - minVal) / range) * trackWidth;
            const maxPos = ((currentMax - minVal) / range) * trackWidth;
            minHandle.style.left = `${minPos}px`;
            maxHandle.style.left = `${maxPos}px`;
            selection.style.left = `${minPos}px`;
            selection.style.width = `${maxPos - minPos}px`;
        };

        setTimeout(updateUI, 50);

        const startDrag = (handle, isMin) => {
            const onMove = (e) => {
                e.preventDefault(); e.stopPropagation();
                const rect = track.getBoundingClientRect();
                const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
                const ratio = Math.max(0, Math.min(1, x / rect.width));
                const value = minVal + ratio * (maxVal - minVal);
                if (isMin) { currentMin = Math.min(value, currentMax - 0.001); minInput.value = currentMin.toFixed(3); }
                else { currentMax = Math.max(value, currentMin + 0.001); maxInput.value = currentMax.toFixed(3); }
                updateUI();
            };
            const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        };

        minHandle.addEventListener('mousedown', (e) => { e.preventDefault(); e.stopPropagation(); startDrag(minHandle, true); });
        maxHandle.addEventListener('mousedown', (e) => { e.preventDefault(); e.stopPropagation(); startDrag(maxHandle, false); });

        minInput.addEventListener('change', (e) => { e.stopPropagation(); currentMin = Math.max(minVal, Math.min(parseFloat(minInput.value), currentMax - 0.001)); minInput.value = currentMin.toFixed(3); updateUI(); });
        minInput.addEventListener('click', (e) => e.stopPropagation());
        maxInput.addEventListener('change', (e) => { e.stopPropagation(); currentMax = Math.min(maxVal, Math.max(parseFloat(maxInput.value), currentMin + 0.001)); maxInput.value = currentMax.toFixed(3); updateUI(); });
        maxInput.addEventListener('click', (e) => e.stopPropagation());

        applyBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await this.applyLocalThreshold(analysisItem, modelId, result, currentMin, currentMax, imgW, imgH);
            if (cancelBtn) cancelBtn.style.display = 'inline-block';
        });

        if (cancelBtn) {
            cancelBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const origUrl = analysisItem.dataset.originalOverlayUrl;
                this.hideLocalAnalysisLayer(modelId);
                this.showLocalAnalysisLayer(modelId, origUrl, imgW, imgH);
                analysisItem.dataset.currentOverlayUrl = origUrl;
                analysisItem.dataset.isBinary = 'false';
                analysisItem.dataset.active = 'true';
                analysisItem.classList.add('active');
                const statusEl = analysisItem.querySelector('.analysis-status');
                if (statusEl) { statusEl.textContent = 'Overlay active'; statusEl.classList.remove('inactive'); statusEl.classList.add('active'); }
                currentMin = minVal; currentMax = maxVal;
                minInput.value = minVal.toFixed(3); maxInput.value = maxVal.toFixed(3);
                updateUI();
                cancelBtn.style.display = 'none';
                this.platform.showNotification('Restored original colormap', 'info');
            });
        }

        container.addEventListener('click', (e) => e.stopPropagation());
    }

    async applyLocalThreshold(analysisItem, modelId, result, minThreshold, maxThreshold, imgW, imgH) {
        this.platform.showLoading('Applying threshold...');
        try {
            const res = await fetch('/api/local/apply-threshold-range', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_dir: this.currentImageDir,
                    algorithm_dir: this.currentAlgoDir,
                    model_id: modelId,
                    min_threshold: minThreshold,
                    max_threshold: maxThreshold,
                    colormap: result.colormap || {}
                })
            });

            if (!res.ok) throw new Error('Threshold failed');
            const data = await res.json();

            if (data.overlay_url) {
                this.hideLocalAnalysisLayer(modelId);
                this.showLocalAnalysisLayer(modelId, data.overlay_url, imgW, imgH, undefined, true);
                analysisItem.dataset.currentOverlayUrl = data.overlay_url;
                analysisItem.dataset.isBinary = 'true';
                analysisItem.dataset.active = 'true';
                analysisItem.classList.add('active');
                const statusEl = analysisItem.querySelector('.analysis-status');
                if (statusEl) { statusEl.textContent = 'Binary mask active'; statusEl.classList.remove('inactive'); statusEl.classList.add('active'); }
            }

            // Register the spectral mask so change-detection can use it.
            if (data.analysis_id && typeof this.platform.registerSlotAnalysis === 'function') {
                const label = result?.colormap?.label || result?.name || modelId;
                this.platform.registerSlotAnalysis({
                    id: data.analysis_id,
                    type: 'spectral',
                    name: `${label} (${minThreshold.toFixed(3)}–${maxThreshold.toFixed(3)})`,
                    hasMask: true,
                });
            }

            this.platform.hideLoading();
            this.platform.showNotification(`Threshold: ${minThreshold.toFixed(3)} ~ ${maxThreshold.toFixed(3)}`, 'success');
        } catch (e) {
            this.platform.hideLoading();
            this.platform.showNotification(`Failed: ${e.message}`, 'error');
        }
    }

    // ===== Local Pixel Value Inspection =====

    toggleLocalPixelInspection(modelId, result, btn) {
        if (this._pixelInspectActive) {
            this.stopLocalPixelInspection();
            return;
        }
        this._pixelInspectActive = true;
        this._pixelInspectModelId = modelId;
        btn.classList.add('active');

        const map = this.currentMode === 'load-image'
            ? window.mapManager?.map
            : this.localMap;

        if (map) {
            map.getContainer().style.cursor = 'crosshair';
            this._pixelClickHandler = async (e) => {
                let row, col;
                if (this.currentMode === 'load-image') {
                    const px = this.latlngToPixel(e.latlng);
                    if (!px) return;
                    row = px.row;
                    col = px.col;
                } else {
                    row = Math.round(e.latlng.lat);
                    col = Math.round(e.latlng.lng);
                }
                try {
                    const res = await fetch('/api/local/get-index-value', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image_dir: this.currentImageDir,
                            algorithm_dir: this.currentAlgoDir,
                            model_id: this._pixelInspectModelId,
                            row: row, col: col
                        })
                    });
                    const data = await res.json();
                    const val = data.value !== undefined ? data.value : data.error || 'N/A';
                    L.popup().setLatLng(e.latlng).setContent(`<b>${result.name}</b><br>Value: ${val}<br>Row: ${row}, Col: ${col}`).openOn(map);
                } catch (err) {
                    console.error('Pixel value error:', err);
                }
            };
            map.on('click', this._pixelClickHandler);
        }
    }

    stopLocalPixelInspection() {
        this._pixelInspectActive = false;
        const map = this.currentMode === 'load-image'
            ? window.mapManager?.map
            : this.localMap;
        if (map) {
            map.getContainer().style.cursor = '';
            if (this._pixelClickHandler) {
                map.off('click', this._pixelClickHandler);
                this._pixelClickHandler = null;
            }
        }
        document.querySelectorAll('.pixel-value-btn.active').forEach(b => b.classList.remove('active'));
    }

    // ===== Add custom viz result to local analysis list =====

    addLocalCustomVizResult(result) {
        const analysisList = document.querySelector('.analysis-list');
        if (!analysisList) return;

        const customItem = analysisList.querySelector('.custom-visualization');
        const imgW = this.preloadedSize.width;
        const imgH = this.preloadedSize.height;

        const resultItem = document.createElement('div');
        resultItem.className = 'analysis-item custom-result';
        resultItem.dataset.modelId = result.custom_id || result.name;
        resultItem.dataset.active = 'false';
        resultItem.innerHTML = `
            <img class="analysis-thumbnail" src="${result.preview_url}" alt="${result.name}" />
            <div class="analysis-info">
                <h4 class="analysis-title">${result.name}</h4>
                <div class="analysis-status inactive">Click to toggle overlay</div>
            </div>
        `;

        resultItem.onclick = () => {
            const isActive = resultItem.dataset.active === 'true';
            const statusEl = resultItem.querySelector('.analysis-status');
            const mid = resultItem.dataset.modelId;

            if (!isActive) {
                this.showLocalAnalysisLayer(mid, result.overlay_url, imgW, imgH);
                resultItem.dataset.active = 'true';
                resultItem.classList.add('active');
                statusEl.textContent = 'Overlay active';
                statusEl.classList.remove('inactive'); statusEl.classList.add('active');
            } else {
                this.hideLocalAnalysisLayer(mid);
                resultItem.dataset.active = 'false';
                resultItem.classList.remove('active');
                statusEl.textContent = 'Click to toggle overlay';
                statusEl.classList.remove('active'); statusEl.classList.add('inactive');
            }
        };

        if (customItem) analysisList.insertBefore(resultItem, customItem);
        else analysisList.appendChild(resultItem);
    }
}
