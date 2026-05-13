/**
 * Change Detection Controller
 *
 * Lives in the right-panel "🔀 Change Detection" tab. Inside the tab,
 * change-detection methods render as `.analysis-item.change-detection-option`
 * cards that mirror the collapsed/expanded "Click to start" pattern used by
 * Target Detection, SAM3, and Spectral Analysis:
 *
 *   collapsed:  thumbnail + title + "Click to start"
 *   expanded:   Time A selector | Time B selector  ─ Run
 *               (result cards stack below, newest on top)
 *
 * Each Run consumes the *binary masks* produced by two prior analyses
 * (target detection, mangrove segmentation, SAM3, or spectral threshold)
 * and renders gained (~A & B) in green, lost (A & ~B) in red, and leaves
 * unchanged pixels fully transparent — same convention every other
 * analysis uses. Dropdowns only list analyses with `hasMask: true`, so
 * the user can't run change detection without first producing a mask.
 *
 * Cloud-mode (GEE) and local-mode (uploaded / local-td) inputs both work;
 * the backend branches on whether the cached entries carry CRS/transform.
 */

class ChangeDetectionController {
    constructor(platformController) {
        this.platform = platformController;
        this.methods = [
            {
                id: 'diff',
                title: 'Magnitude Diff |B − A|',
                description: 'Per-pixel absolute difference; threshold the magnitude to get a change mask.',
                gradient: 'linear-gradient(135deg, #e53935 0%, #fb8c00 100%)',
                icon: '🔀',
            },
            // Future methods add another entry here → another expandable box.
        ];
        this.results = [];               // { id, method, a, b, detection_result, stats, visible, layerId, aName, bName }
        this.overlayLayers = {};         // resultId → layerId (for local hide/show parity)

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this._wire());
        } else {
            this._wire();
        }

        window.addEventListener('slot:switched', () => this._refreshAllDropdowns());

        // Layer-panel groups (intersection / union / exclusion of binary
        // masks) are eligible inputs for change detection; refresh whenever
        // the registry changes so newly formed groups show up immediately.
        window.addEventListener('mask:registry-changed', () => this._refreshAllDropdowns());

        // Keep dropdowns + result cards in sync when the user renames a
        // layer via the layer control panel.
        window.addEventListener('slot-analysis:renamed', (e) => {
            this._handleAnalysisRenamed(e.detail || {});
        });
    }

    /**
     * Update (a) A/B dropdown option text, (b) aName/bName on every result
     * card whose source analysis id matches, (c) compare-mode layer panel
     * in case it's visible.
     */
    _handleAnalysisRenamed({ layerId, newName }) {
        if (!layerId || !newName) return;

        // (a) Re-populate each expanded dropdown from fresh platform state.
        this._refreshAllDropdowns();

        // (b) Patch cached aName/bName on results that reference this id
        //     and rewrite the pieces of the DOM that display them.
        for (const entry of this.results) {
            let changed = false;
            if (entry.a === layerId) { entry.aName = newName; changed = true; }
            if (entry.b === layerId) { entry.bName = newName; changed = true; }
            if (!changed) continue;

            const itemEl = document.querySelector(
                `.cd-result-item[data-result-id="${CSS.escape(entry.id)}"]`
            );
            if (!itemEl) continue;
            const nameEl = itemEl.querySelector('.td-result-name');
            if (nameEl) {
                nameEl.textContent = `${this._shortTail(entry.aName, 14)} ↔ ${this._shortTail(entry.bName, 14)}`;
                nameEl.title = `${entry.aName} ↔ ${entry.bName}`;
            }
            const pairEl = itemEl.querySelector('.cd-result-pair');
            if (pairEl) {
                pairEl.innerHTML =
                    `A: <b>${this._escape(entry.aName)}</b><br/>` +
                    `B: <b>${this._escape(entry.bName)}</b>`;
            }
        }

        // (c) If compare mode is open, re-render its layer lists so the new
        //     name shows up immediately in the Time A / Time B sections.
        const dm = this.platform?.dualMapController;
        if (dm?.isActive && typeof dm._renderAssignmentUI === 'function') {
            dm._renderAssignmentUI();
        }
    }

    // ========== Local vs cloud mode (parity with other controllers) ==========

    isLocalMode() {
        const li = this.platform.localImage;
        return !!li?.localMapActive || !!li?.isUploadedImageActive;
    }

    // ========== Initial wiring ==========

    _wire() {
        this.tabBtn = document.getElementById('change-detection-tab-btn');
        this.tabContent = document.getElementById('right-change-detection-tab-content');
        this.listEl = document.getElementById('cd-analysis-list');

        if (!this.tabBtn || !this.tabContent || !this.listEl) {
            console.warn('[change-detection] DOM not ready yet; retrying in 400ms');
            setTimeout(() => this._wire(), 400);
            return;
        }

        this._renderMethodItems();

        this.tabBtn.addEventListener('click', () => {
            if (this.tabBtn.disabled) return;
            this._refreshAllDropdowns();
        });

        this.updateTabAvailability();
    }

    /**
     * Called by PlatformController._updateSlotToolbar whenever slot state
     * changes. Tab is enabled once both Time A and Time B have at least one
     * analysis whose binary mask is ready (TD/SAM3 always; mangrove +
     * spectral once the user applies a threshold).
     */
    updateTabAvailability() {
        if (!this.tabBtn) return;
        const aMasks = (this.platform.getSlotAnalyses?.('A') || []).filter(a => a.hasMask).length
            + this._getMaskGroupsForSlot('A').length;
        const bMasks = (this.platform.getSlotAnalyses?.('B') || []).filter(a => a.hasMask).length
            + this._getMaskGroupsForSlot('B').length;
        const ready = aMasks > 0 && bMasks > 0;
        this.tabBtn.disabled = !ready;
        this.tabBtn.title = ready
            ? 'Compare binary masks between Time A and Time B'
            : 'Produce a binary mask in each slot (target detection, SAM3, or apply a threshold) to enable';
        if (!ready && this.tabBtn.classList.contains('active')) {
            const analysisBtn = document.querySelector('[data-right-tab="analysis"]');
            analysisBtn?.click();
        }
    }

    // ========== Method items (collapsed "Click to start" cards) ==========

    _renderMethodItems() {
        this.listEl.innerHTML = '';
        for (const m of this.methods) {
            const item = document.createElement('div');
            item.className = `analysis-item change-detection-option cd-method-${m.id}`;
            item.dataset.methodId = m.id;
            item.dataset.active = 'false';
            item.innerHTML = `
                <div class="analysis-thumbnail custom-placeholder" style="background: ${m.gradient};">
                    <div class="custom-icon">${m.icon}</div>
                </div>
                <div class="analysis-info">
                    <h4 class="analysis-title">${this._escape(m.title)}</h4>
                    <div class="analysis-status inactive">Click to start</div>
                </div>
            `;
            item.onclick = (e) => {
                if (e.target.closest('.cd-ui')) return;
                this._handleMethodClick(m.id);
            };
            this.listEl.appendChild(item);
        }
    }

    _handleMethodClick(methodId) {
        const item = this.listEl.querySelector(`.analysis-item[data-method-id="${methodId}"]`);
        if (!item) return;
        const existing = item.querySelector('.cd-ui');
        if (existing) {
            this._hideMethodUI(methodId);
        } else {
            this._showMethodUI(methodId);
        }
    }

    _showMethodUI(methodId) {
        const item = this.listEl.querySelector(`.analysis-item[data-method-id="${methodId}"]`);
        if (!item) return;
        item.querySelector('.cd-ui')?.remove();

        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = 'none';

        const ui = document.createElement('div');
        ui.className = 'cd-ui';
        ui.innerHTML = `
            <div class="cd-setup-row">
                <div class="cd-field">
                    <label>Time A</label>
                    <select class="cd-select-a"></select>
                </div>
                <div class="cd-field">
                    <label>Time B</label>
                    <select class="cd-select-b"></select>
                </div>
            </div>
            <div class="cd-action-row">
                <button class="cd-run-btn control-btn primary small">Run</button>
                <span class="cd-status"></span>
            </div>
            <div class="cd-results-list"></div>
        `;

        item.appendChild(ui);
        item.classList.add('expanded');
        item.dataset.active = 'true';

        // Populate A/B dropdowns from the slot registry.
        this._populateDropdowns(ui);

        // Wire controls.
        const runBtn = ui.querySelector('.cd-run-btn');
        runBtn.onclick = (e) => {
            e.stopPropagation();
            this._submit(methodId, ui);
        };

        // Re-render any existing results for this method.
        const myResults = this.results.filter(r => r.method === methodId);
        const resultsListEl = ui.querySelector('.cd-results-list');
        for (let i = myResults.length - 1; i >= 0; i--) {
            this._renderResultItem(myResults[i], resultsListEl);
        }
    }

    _hideMethodUI(methodId) {
        const item = this.listEl.querySelector(`.analysis-item[data-method-id="${methodId}"]`);
        if (!item) return;
        item.querySelector('.cd-ui')?.remove();
        item.classList.remove('expanded');
        item.dataset.active = 'false';
        const infoEl = item.querySelector('.analysis-info');
        if (infoEl) infoEl.style.display = '';
    }

    // ========== A/B dropdowns ==========

    _populateDropdowns(uiEl) {
        const slotA = (this.platform.getSlotAnalyses?.('A') || []).filter(a => a.hasMask);
        const slotB = (this.platform.getSlotAnalyses?.('B') || []).filter(a => a.hasMask);
        const groupsA = this._getMaskGroupsForSlot('A');
        const groupsB = this._getMaskGroupsForSlot('B');
        const selA = uiEl.querySelector('.cd-select-a');
        const selB = uiEl.querySelector('.cd-select-b');
        this._fillSelect(selA, slotA, groupsA);
        this._fillSelect(selB, slotB, groupsB);
        // Stash the resolved groups so _submit can rebuild the components
        // payload from the chosen <option value="group:idx">.
        uiEl._cdGroups = { A: groupsA, B: groupsB };
    }

    /**
     * Read mask-groups (composited via intersection / union / exclusion in
     * the layer panel) from the slot's registry. Returns one descriptor per
     * group with `items` and `operators` matching the panel's model so the
     * backend can re-compose from the original cached binary masks.
     *
     * Translation note: layer-panel `entry.id` is the *layer* id used on
     * the map (e.g. `td-td-…`), not the analysis cache id (`td-…`). The
     * change-detection backend looks results up by cache id, so each item
     * is mapped through `_resolveCacheIdForLayer()` before being returned.
     * Items that can't be resolved (e.g. stale entries, or a binary layer
     * that was never registered as a slot analysis) are dropped — and if
     * fewer than 2 items survive, the group itself is dropped.
     */
    _getMaskGroupsForSlot(slotId) {
        const panel = window.layerControlPanel;
        if (!panel?._slotLayers) return [];
        const slotRegs = panel._slotLayers[slotId] || {};
        const slotAnalyses = this.platform.getSlotAnalyses?.(slotId) || [];
        const result = [];
        for (const tab of Object.keys(slotRegs)) {
            for (const row of slotRegs[tab] || []) {
                if (row?.kind !== 'group') continue;
                const rawItems = (row.items || []).filter(
                    e => e?.id && e.isBinary && !String(e.id).startsWith('chg_')
                );
                if (rawItems.length < 2) continue;

                // Map each panel item to its underlying analysis cache id;
                // also track which raw items survived so we can drop the
                // matching operator slots in step.
                const keptItems = [];
                const keptOperatorMask = [];
                for (let k = 0; k < rawItems.length; k++) {
                    const e = rawItems[k];
                    const cacheId = this._resolveCacheIdForLayer(e.id, slotAnalyses);
                    if (!cacheId) {
                        console.warn('[change-detection] dropping group item with no slot-analysis match', e.id);
                        keptOperatorMask.push(false);
                        continue;
                    }
                    keptItems.push({ id: cacheId, name: e.name, type: e.type });
                    keptOperatorMask.push(true);
                }
                if (keptItems.length < 2) continue;

                // Operators correspond to joins between items[k] and items[k+1].
                // After dropping items, keep operator[i] only when both
                // items[i] and items[i+1] survived. As a fallback for ragged
                // cases, pad with 'inc'.
                const rawOps = row.operators || [];
                const ops = [];
                for (let k = 1; k < rawItems.length; k++) {
                    if (keptOperatorMask[k - 1] && keptOperatorMask[k]) {
                        ops.push(rawOps[k - 1] || 'inc');
                    }
                }
                while (ops.length < keptItems.length - 1) ops.push('inc');
                ops.length = keptItems.length - 1;

                result.push({ items: keptItems, operators: ops });
            }
        }
        return result;
    }

    /**
     * Translate a layer-panel layer id (e.g. `td-td-1778…`, `ms-ms-…`,
     * `sam3-text-…`) to the analysis cache id stored in the slot registry.
     * Strategy, in order:
     *   1. exact match against a slot analysis id → use as-is.
     *   2. suffix match: pick the longest slot-analysis id that the layer
     *      id ends with (handles arbitrary `<prefix>-<cacheId>` shapes).
     *   3. give up → return null.
     */
    _resolveCacheIdForLayer(layerId, slotAnalyses) {
        if (!layerId) return null;
        if (slotAnalyses.some(a => a.id === layerId)) return layerId;
        let best = null;
        for (const a of slotAnalyses) {
            if (!a.id) continue;
            if (layerId === a.id || layerId.endsWith('-' + a.id) || layerId.endsWith(a.id)) {
                if (best === null || a.id.length > best.length) best = a.id;
            }
        }
        return best;
    }

    _groupLabel(group) {
        const sep = { inc: ' ∩ ', exc: ' − ', add: ' ∪ ' };
        const names = group.items.map(it => this._shortTail(it.name, 12));
        let out = names[0];
        for (let k = 1; k < names.length; k++) {
            out += (sep[group.operators[k - 1]] || ' ? ') + names[k];
        }
        return `Group: ${out}`;
    }

    _fillSelect(sel, items, groups = []) {
        if (!sel) return;
        const prev = sel.value;
        if (items.length === 0 && groups.length === 0) {
            sel.innerHTML = '<option value="" disabled selected>No binary masks — apply a threshold first</option>';
            return;
        }
        const groupOpts = groups.map((g, idx) =>
            `<option value="group:${idx}" data-type="group">${this._escape(this._groupLabel(g))}</option>`
        );
        const itemOpts = items.map(a => {
            const typeTag = a.type ? `[${this._typeBadge(a.type)}] ` : '';
            return `<option value="${this._escapeAttr(a.id)}" data-type="${this._escapeAttr(a.type || '')}">${this._escape(typeTag + a.name)}</option>`;
        });
        sel.innerHTML = [...groupOpts, ...itemOpts].join('');
        // Restore prior selection when still valid.
        if (prev) {
            if (prev.startsWith('group:')) {
                const idx = parseInt(prev.slice(6), 10);
                if (Number.isFinite(idx) && idx < groups.length) sel.value = prev;
            } else if (items.some(a => a.id === prev)) {
                sel.value = prev;
            }
        }
    }

    _typeBadge(type) {
        switch (type) {
            case 'detection': return 'TD';
            case 'segmentation': return 'MS';
            case 'sam2': return 'SAM3';
            case 'sam3': return 'SAM3';
            case 'spectral': return 'SA';
            default: return type;
        }
    }

    _refreshAllDropdowns() {
        if (!this.listEl) return;
        this.listEl.querySelectorAll('.cd-ui').forEach(ui => this._populateDropdowns(ui));
    }

    // ========== Run / Poll ==========

    async _submit(methodId, uiEl) {
        const selA = uiEl.querySelector('.cd-select-a');
        const selB = uiEl.querySelector('.cd-select-b');
        const rawA = selA?.value;
        const rawB = selB?.value;
        const runBtn = uiEl.querySelector('.cd-run-btn');
        const statusEl = uiEl.querySelector('.cd-status');

        if (!rawA || !rawB) {
            this._setStatus(statusEl, 'Select a layer for each slot.', 'error');
            return;
        }
        if (rawA === rawB) {
            this._setStatus(statusEl, 'Time A and Time B must be different layers.', 'error');
            return;
        }

        const sideA = this._resolveSelection('A', rawA, uiEl);
        const sideB = this._resolveSelection('B', rawB, uiEl);
        if (!sideA || !sideB) {
            this._setStatus(statusEl, 'Internal: selected option no longer exists. Reopen the panel.', 'error');
            return;
        }

        const bbox = this.isLocalMode()
            ? null
            : (this.platform.processedBbox
                || (window.mapManager?.getCurrentBounds
                    ? window.mapManager.getCurrentBounds()
                    : null));

        runBtn.disabled = true;
        this._setStatus(statusEl, 'Submitting…');

        const aName = sideA.label;
        const bName = sideB.label;
        // Use the first component id of a group as the slot-registry lookup
        // key so cd-result-pair rename events can still patch the card.
        const idA = sideA.kind === 'single' ? sideA.id : sideA.components[0]?.id;
        const idB = sideB.kind === 'single' ? sideB.id : sideB.components[0]?.id;

        const run = async () => {
            const payload = { bbox };
            if (sideA.kind === 'single') payload.result_a_id = sideA.id;
            else payload.result_a_group = { items: sideA.components.map(c => c.id), operators: sideA.operators };
            if (sideB.kind === 'single') payload.result_b_id = sideB.id;
            else payload.result_b_group = { items: sideB.components.map(c => c.id), operators: sideB.operators };

            const resp = await fetch('/api/change-detection/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                throw new Error(data.detail || `HTTP ${resp.status}`);
            }
            const result = await resp.json();
            this._setStatus(statusEl, 'Complete.');
            this._addResult(result, { methodId, uiEl, idA, idB, aName, bName, runBtn, statusEl });
        };

        try {
            const platform = this.platform || window.app;
            if (platform && typeof platform.withLoading === 'function') {
                await platform.withLoading('Computing change detection...', run);
            } else {
                await run();
            }
            runBtn.disabled = false;
        } catch (err) {
            console.error('[change-detection] submit failed', err);
            this._setStatus(statusEl, err.message || String(err), 'error');
            runBtn.disabled = false;
        }
    }

    _setStatus(el, text, kind) {
        if (!el) return;
        el.textContent = text;
        el.classList.toggle('error', kind === 'error');
    }

    _findAnalysisName(slotId, analysisId) {
        const list = this.platform.getSlotAnalyses?.(slotId) || [];
        return list.find(a => a.id === analysisId)?.name;
    }

    /**
     * Map a <select> raw value to either a single-analysis side or a
     * composed-group side. Group values look like "group:<idx>" where idx
     * indexes the per-slot list stashed on the UI element.
     */
    _resolveSelection(slotId, rawValue, uiEl) {
        if (rawValue && rawValue.startsWith('group:')) {
            const idx = parseInt(rawValue.slice(6), 10);
            const group = uiEl?._cdGroups?.[slotId]?.[idx];
            if (!group) return null;
            return {
                kind: 'group',
                components: group.items,
                operators: group.operators,
                label: this._groupLabel(group).replace(/^Group:\s*/, ''),
            };
        }
        return {
            kind: 'single',
            id: rawValue,
            label: this._findAnalysisName(slotId, rawValue) || rawValue,
        };
    }

    // ========== Result lifecycle ==========

    _addResult(result, ctx) {
        const id = result.change_detection_id;
        if (!id) return;

        const entry = {
            id,
            method: ctx.methodId,
            layerId: `chg_${id}`,
            a: ctx.idA,
            b: ctx.idB,
            aName: ctx.aName,
            bName: ctx.bName,
            stats: result.stats || {},
            detection_result: result.detection_result,   // gained/lost overlay
            visible: true,
            slotIdAtRun: this.platform.currentSlot,
        };
        this.results.unshift(entry);
        this._showOverlay(entry, entry.detection_result);

        const resultsListEl = ctx.uiEl.querySelector('.cd-results-list');
        this._renderResultItem(entry, resultsListEl);
    }

    _renderResultItem(entry, resultsListEl) {
        if (!resultsListEl) return;
        if (resultsListEl.querySelector(`[data-result-id="${CSS.escape(entry.id)}"]`)) return;

        const stats = entry.stats || {};
        const gainedPct = (stats.gained_pct ?? 0).toFixed(2);
        const lostPct = (stats.lost_pct ?? 0).toFixed(2);
        const unchangedPct = (stats.unchanged_pct ?? 0).toFixed(2);

        const item = document.createElement('div');
        item.className = 'td-result-item cd-result-item';
        item.dataset.resultId = entry.id;
        item.dataset.visible = 'true';

        item.innerHTML = `
            <div class="td-result-header">
                <img class="td-result-thumb" src="${this._escapeAttr(entry.detection_result?.preview_url || '')}" alt="Change mask" />
                <div class="td-result-info">
                    <span class="td-result-name" title="${this._escape(entry.aName)} ↔ ${this._escape(entry.bName)}">${this._escape(this._shortTail(entry.aName, 14))} ↔ ${this._escape(this._shortTail(entry.bName, 14))}</span>
                    <span class="td-result-status">Gained ${gainedPct}% · Lost ${lostPct}%</span>
                </div>
                <div class="td-result-actions">
                    <button class="td-result-remove-btn" title="Remove">✕</button>
                </div>
            </div>
            <div class="cd-result-pair">
                A: <b>${this._escape(entry.aName)}</b><br/>
                B: <b>${this._escape(entry.bName)}</b>
            </div>
            <div class="cd-stats">
                <div class="cd-stat gained"><span class="cd-stat-swatch" style="background:#1bbf4d;"></span>Gained<div class="cd-stat-value">${gainedPct}%</div></div>
                <div class="cd-stat lost"><span class="cd-stat-swatch" style="background:#dc2626;"></span>Lost<div class="cd-stat-value">${lostPct}%</div></div>
                <div class="cd-stat unchanged">Unchanged<div class="cd-stat-value">${unchangedPct}%</div></div>
            </div>
        `;

        resultsListEl.prepend(item);

        item.querySelector('.td-result-header').onclick = (e) => {
            if (e.target.closest('.td-result-actions')) return;
            e.stopPropagation();
            this._toggleOverlay(entry.id);
        };
        item.querySelector('.td-result-remove-btn').onclick = (e) => {
            e.stopPropagation();
            this._removeResult(entry.id);
        };
    }

    // ========== Overlay management (cloud + local parity) ==========

    _showOverlay(entry, overlayData) {
        if (!overlayData?.overlay_url) {
            console.warn('[CD] _showOverlay skipped: overlay_url missing', overlayData);
            return;
        }
        this._hideOverlay(entry);

        const meta = overlayData.overlay_meta || {};
        const displayName = `Change · ${this._shortTail(entry.aName, 10)}↔${this._shortTail(entry.bName, 10)}`;

        // Prefer the backend's explicit mode — it already branched based on
        // whether the cached inputs carry CRS/transform. Only fall back to
        // isLocalMode() when meta didn't tell us anything.
        if (meta.mode === 'local') {
            const li = this.platform.localImage;
            li?.showLocalAnalysisLayer?.(
                entry.layerId,
                overlayData.overlay_url,
                meta.width,
                meta.height,
                displayName,
                true, // change-detection result is a binary mask
            );
        } else if (meta.mode === 'geo' && meta.bounds && window.mapManager?.layers?.showAnalysisLayer) {
            const [s, w, n, e] = meta.bounds;
            window.mapManager.layers.showAnalysisLayer(
                entry.layerId,
                overlayData.overlay_url,
                displayName,
                [[s, w], [n, e]],
                true,
            );
        } else if (meta.bounds && window.mapManager?.layers?.showAnalysisLayer) {
            // Legacy fallback when meta.mode is absent.
            const [s, w, n, e] = meta.bounds;
            window.mapManager.layers.showAnalysisLayer(
                entry.layerId, overlayData.overlay_url, displayName, [[s, w], [n, e]], true,
            );
        } else {
            console.warn('[CD] _showOverlay could not render: unsupported overlay_meta', meta);
            return;
        }
        entry.visible = true;
        this.overlayLayers[entry.id] = entry.layerId;
        this._updateItemVisibleState(entry.id, true);
    }

    _hideOverlay(entry) {
        if (!entry) return;
        // Call both paths unconditionally — each is a no-op when the layer
        // isn't there, and this protects us when the overlay was added via
        // one path (e.g. meta.mode='geo' → mapManager) while isLocalMode()
        // returns true (e.g. load-image mode).
        try { this.platform.localImage?.hideLocalAnalysisLayer?.(entry.layerId); } catch (e) {}
        try { window.mapManager?.layers?.hideAnalysisLayer?.(entry.layerId); } catch (e) {}
        entry.visible = false;
        delete this.overlayLayers[entry.id];
        this._updateItemVisibleState(entry.id, false);
    }

    _toggleOverlay(resultId) {
        const entry = this.results.find(r => r.id === resultId);
        if (!entry) return;
        if (entry.visible) {
            this._hideOverlay(entry);
        } else {
            this._showOverlay(entry, entry.detection_result);
        }
    }

    _removeResult(resultId) {
        const entry = this.results.find(r => r.id === resultId);
        if (!entry) return;
        if (entry.visible) this._hideOverlay(entry);
        this.results = this.results.filter(r => r.id !== resultId);
        const el = document.querySelector(`.cd-result-item[data-result-id="${CSS.escape(resultId)}"]`);
        if (el) el.remove();
    }

    _updateItemVisibleState(resultId, visible) {
        const el = document.querySelector(`.cd-result-item[data-result-id="${CSS.escape(resultId)}"]`);
        if (!el) return;
        el.dataset.visible = visible ? 'true' : 'false';
        const status = el.querySelector('.td-result-status');
        if (status && !visible) status.textContent = 'Hidden — click to show';
    }

    // ========== Compare-mode integration ==========

    /**
     * Flatten CD results into layer-entry shape that DualMapController
     * understands. Includes the overlay metadata each side of compare mode
     * needs to reconstruct a Leaflet layer locally (CD layers aren't
     * registered with LayerControlPanel, so compare mode can't find them
     * through `getAllAnalysisLayersBySlot`).
     *
     * Returns: [{id, name, type:'change-detection', overlayData, ...}]
     */
    listResultsForCompareMode() {
        return this.results.map(r => {
            const data = r.detection_result;
            return {
                id: r.layerId,                                 // e.g. "chg_chg-123"
                name: `Change · ${this._shortTail(r.aName, 10)}↔${this._shortTail(r.bName, 10)}`,
                type: 'change-detection',
                resultId: r.id,
                overlayUrl: data?.overlay_url,
                overlayMeta: data?.overlay_meta,
                visible: r.visible,
            };
        });
    }

    // ========== Helpers ==========

    _escape(s) {
        return String(s).replace(/[&<>"']/g, c => (
            {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
        ));
    }
    _escapeAttr(s) { return this._escape(s); }
    _shortTail(s, n) {
        s = String(s || '');
        return s.length > n ? '…' + s.slice(-(n - 2)) : s;
    }
}

if (typeof window !== 'undefined') {
    window.ChangeDetectionController = ChangeDetectionController;
}
