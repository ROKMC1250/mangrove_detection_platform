/**
 * Layer Control Panel Module
 * Shows active overlay layers with drag-to-reorder, visibility toggle, and rename.
 *
 * Slot-scoped: each Time-A / Time-B slot maintains its own per-tab registry,
 * so switching slots swaps the list (no leaking between slots).
 * The whole row is draggable (not just the handle). Double-click the name
 * to rename — the new label propagates to `platform._slots[slot].analyses`
 * so the Change Detection picker shows the user's custom name.
 */

class LayerControlPanel {
    constructor(mapManager) {
        this.mapManager = mapManager;
        this.currentTab = 'search';

        // Active slot — swapped via `slot:switched` event.
        this.currentSlot = this._detectCurrentSlot();

        // Per-slot, per-tab registries (ordered arrays, bottom-to-top).
        // Layers are scoped to the slot that was active when they were added.
        this._slotLayers = {
            A: { 'search': [], 'local-image': [] },
            B: { 'search': [], 'local-image': [] },
        };

        this.sectionEl = document.getElementById('layer-control-section');
        this.listEl = document.getElementById('layer-list');
        this.toggleBtn = document.getElementById('layer-control-toggle');
        this.collapsed = false;
        this._dragSrcIndex = null;

        if (!this.sectionEl || !this.listEl) {
            console.warn('[LayerControlPanel] DOM elements not found, will retry.');
            setTimeout(() => {
                this.sectionEl = document.getElementById('layer-control-section');
                this.listEl = document.getElementById('layer-list');
                this.toggleBtn = document.getElementById('layer-control-toggle');
                if (this.toggleBtn) {
                    this.toggleBtn.addEventListener('click', () => this._toggleCollapse());
                }
                console.log('[LayerControlPanel] Retry DOM lookup:', !!this.sectionEl, !!this.listEl);
            }, 500);
        }

        this._bindEvents();

        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this._toggleCollapse());
        }

        console.log('[LayerControlPanel] Initialized, slot=' + this.currentSlot);
    }

    // ========== Accessors ==========

    _detectCurrentSlot() {
        // PlatformController may not be ready at construction time — default to 'A'.
        const s = window.platform?.currentSlot;
        return (s === 'A' || s === 'B') ? s : 'A';
    }

    /**
     * Legacy-compat shim: existing code in this class reads/writes
     * `this.layers[tab]`. Keep that API working by proxying to the active
     * slot's per-tab registry.
     */
    get layers() {
        return this._slotLayers[this.currentSlot];
    }
    set layers(v) {
        // Preserve assignments like `this.layers[tab] = []` by writing into
        // the current slot bucket. Assigning the whole object is uncommon;
        // if it happens, replace the current slot's buckets.
        if (v && typeof v === 'object' && !Array.isArray(v)) {
            this._slotLayers[this.currentSlot] = v;
        }
    }

    _bindEvents() {
        // Multi-level dragover so the user can drop *anywhere* — even
        // outside the panel onto the map. Tiered priority:
        //   1. list/section: precise drop index from cursor Y vs item midpoints
        //   2. document: fallback when cursor leaves the panel — clamps to
        //      top (cursor above section) or bottom (cursor below section)
        // Item-level handlers also fire but the list-level handler is
        // authoritative for index calculation.
        const wireListLevel = () => {
            const list = this.listEl;
            const section = this.sectionEl;
            if (list) {
                list.addEventListener('dragover', (e) => this._onListDragOver(e));
                list.addEventListener('drop',     (e) => this._onListDrop(e));
                list.addEventListener('dragleave',(e) => this._onListDragLeave(e));
            }
            if (section) {
                section.addEventListener('dragover', (e) => this._onListDragOver(e));
                section.addEventListener('drop',     (e) => this._onListDrop(e));
            }
        };
        if (this.listEl || this.sectionEl) wireListLevel();
        else setTimeout(wireListLevel, 600);

        // Document-level fallback. Active only while a layer-row drag is in
        // progress (`_dragSrcIndex !== null`), so we don't intercept drag
        // events from unrelated UI (band chips, file uploads).
        document.addEventListener('dragover', (e) => this._onDocDragOver(e));
        document.addEventListener('drop',     (e) => this._onDocDrop(e));

        window.addEventListener('layer:added', (e) => {
            try { this._onLayerAdded(e.detail); } catch (err) { console.error('[LayerControlPanel] layer:added error:', err); }
        });
        window.addEventListener('layer:removed', (e) => {
            try { this._onLayerRemoved(e.detail); } catch (err) { console.error('[LayerControlPanel] layer:removed error:', err); }
        });
        window.addEventListener('layers:cleared', (e) => {
            try { this._onLayersCleared(e.detail); } catch (err) { console.error('[LayerControlPanel] layers:cleared error:', err); }
        });
        window.addEventListener('tab:switched', (e) => {
            try { this._onTabSwitched(e.detail); } catch (err) { console.error('[LayerControlPanel] tab:switched error:', err); }
        });
        // Time A / Time B slot handoff — MapLayers fires this every switch.
        window.addEventListener('slot:switched', (e) => {
            try { this._onSlotSwitched(e.detail); } catch (err) { console.error('[LayerControlPanel] slot:switched error:', err); }
        });
    }

    _getActiveTab() {
        return this.currentTab;
    }

    // ========== Row model helpers ==========
    //
    // _slotLayers[slot][tab] is an array of "rows". Each row is one of:
    //   { kind: 'single', entry }
    //   { kind: 'group',  id, items: entry[], operators: string[] }
    // operators.length === items.length - 1; each operator is 'inc'|'add'|'exc'.
    // Group items are always isBinary === true. Mask composition operates on
    // groups; standalone (single) rows never participate.

    _wrapSingle(entry) {
        return { kind: 'single', entry };
    }

    _isGroup(row) {
        return row && row.kind === 'group';
    }

    _isSingle(row) {
        return row && row.kind === 'single';
    }

    _flattenRows(rows) {
        const out = [];
        for (const row of (rows || [])) {
            if (this._isSingle(row)) out.push(row.entry);
            else if (this._isGroup(row)) out.push(...(row.items || []));
        }
        return out;
    }

    _findRowAndIndex(rows, entryId) {
        for (let r = 0; r < rows.length; r++) {
            const row = rows[r];
            if (this._isSingle(row) && row.entry?.id === entryId) {
                return { rowIdx: r, itemIdx: -1 };
            }
            if (this._isGroup(row)) {
                const idx = (row.items || []).findIndex(e => e?.id === entryId);
                if (idx !== -1) return { rowIdx: r, itemIdx: idx };
            }
        }
        return null;
    }

    _newGroupId() {
        return 'mg_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    }

    /**
     * Remove an item from a group row at (rowIdx, itemIdx). Adjacent operator
     * is dropped (left-side preferred). If the group ends up with a single
     * item, dissolve it into a single row in place.
     * Returns the removed entry (so callers can re-insert as a single row).
     */
    _detachFromGroup(rows, rowIdx, itemIdx) {
        const row = rows[rowIdx];
        if (!this._isGroup(row)) return null;
        const removed = row.items.splice(itemIdx, 1)[0] || null;
        // Drop the operator that was paired with this item:
        //   if removing items[0], drop operators[0]
        //   else drop operators[itemIdx - 1]
        const opIdx = itemIdx === 0 ? 0 : itemIdx - 1;
        if (row.operators.length > 0 && opIdx < row.operators.length) {
            row.operators.splice(opIdx, 1);
        }
        if (row.items.length === 1) {
            // Dissolve to single row in place.
            rows[rowIdx] = this._wrapSingle(row.items[0]);
        } else if (row.items.length === 0) {
            rows.splice(rowIdx, 1);
        }
        return removed;
    }

    _onLayerAdded(detail) {
        if (!detail || !detail.id) return;
        // TEMP diagnostic — fires every time any layer is added; tells us
        // whether the producer passed isBinary correctly.
        console.log('[mask-merge] layer:added', {
            id: detail.id, type: detail.type, isBinary: detail.isBinary, name: detail.name,
        });

        const tab = this._getActiveTab();
        const registry = this._slotLayers[this.currentSlot][tab];

        // Remove existing entry first so re-added layers move to top. Preserve
        // the user's custom name across re-adds (happens on threshold tweaks).
        // The existing entry might be a single row or an item inside a group.
        let preservedName = null;
        const found = this._findRowAndIndex(registry, detail.id);
        if (found) {
            const row = registry[found.rowIdx];
            if (this._isSingle(row)) {
                preservedName = row.entry.userRenamed ? row.entry.name : null;
                registry.splice(found.rowIdx, 1);
            } else if (this._isGroup(row)) {
                const item = row.items[found.itemIdx];
                preservedName = item.userRenamed ? item.name : null;
                // Re-add of an in-group item: detach it and re-add as a fresh
                // single row at the top. Group integrity is preserved.
                this._detachFromGroup(registry, found.rowIdx, found.itemIdx);
                // If that detach dissolved the group → single, the survivor
                // was off-map (composite was representing it). Put it back.
                const dissolved = registry[found.rowIdx];
                if (this._isSingle(dissolved)) {
                    this._setEntryOnMap(dissolved.entry, dissolved.entry.visible !== false);
                }
            }
        }

        const type = detail.type || 'analysis';
        const entry = {
            id: detail.id,
            type,
            name: preservedName || detail.name || detail.id,
            userRenamed: !!preservedName,
            leafletLayer: detail.layer,
            visible: true,
            slotId: this.currentSlot,
            // Whether this layer's overlay is a binary mask (post-threshold,
            // segmentation, change-detection, etc). Only binary layers can
            // participate in mask groups.
            isBinary: detail.isBinary === true,
            // Legacy field kept for compositor read-path; new UI uses groups
            // and ignores this. Always 'off' under the new model.
            maskRole: 'off',
        };
        registry.push(this._wrapSingle(entry));

        this._render();
        // Apply z-order after short delay so DOM elements are ready.
        setTimeout(() => this._applyZOrder(), 100);
        this._updateCompareButtons();
        this._notifyMaskRegistry();

        if (this.listEl) this.listEl.scrollTop = 0;
    }

    _onLayerRemoved(detail) {
        if (!detail || !detail.id) return;

        // Search across both tabs of the CURRENT slot only — a remove event
        // only applies to the active slot because inactive-slot overlays
        // have already been detached from the map.
        const slotRegs = this._slotLayers[this.currentSlot];
        for (const tab of Object.keys(slotRegs)) {
            const registry = slotRegs[tab];
            const found = this._findRowAndIndex(registry, detail.id);
            if (!found) continue;
            const row = registry[found.rowIdx];
            if (this._isSingle(row)) {
                registry.splice(found.rowIdx, 1);
            } else if (this._isGroup(row)) {
                this._detachFromGroup(registry, found.rowIdx, found.itemIdx);
                // Group dissolved to single → survivor goes back on map.
                const dissolved = registry[found.rowIdx];
                if (this._isSingle(dissolved)) {
                    this._setEntryOnMap(dissolved.entry, dissolved.entry.visible !== false);
                }
            }
            break;
        }
        this._render();
        this._updateCompareButtons();
        this._notifyMaskRegistry();
    }

    _onLayersCleared(detail) {
        // `layers:cleared` fires from `clearAllAnalysisLayers` which nukes
        // both slots' stashes — so we wipe entries in BOTH slots.
        const filterType = detail && detail.type;
        for (const slot of ['A', 'B']) {
            const regs = this._slotLayers[slot];
            for (const tab of Object.keys(regs)) {
                if (!filterType) {
                    regs[tab] = [];
                    continue;
                }
                const next = [];
                for (const row of regs[tab]) {
                    if (this._isSingle(row)) {
                        if (row.entry.type !== filterType) next.push(row);
                        continue;
                    }
                    if (this._isGroup(row)) {
                        const keptItems = row.items.filter(e => e.type !== filterType);
                        if (keptItems.length === row.items.length) {
                            next.push(row);
                        } else if (keptItems.length >= 2) {
                            // Drop operators paired with removed items; keep
                            // first (n-1) operators as a coarse heuristic.
                            row.items = keptItems;
                            row.operators = row.operators.slice(0, keptItems.length - 1);
                            next.push(row);
                        } else if (keptItems.length === 1) {
                            next.push(this._wrapSingle(keptItems[0]));
                        }
                    }
                }
                regs[tab] = next;
            }
        }
        this._render();
        this._updateCompareButtons();
        this._notifyMaskRegistry();
    }

    _onTabSwitched(detail) {
        if (!detail || !detail.tab) return;
        this.currentTab = detail.tab;
        this._render();
        this._updateCompareButtons();
        this._notifyMaskRegistry();
    }

    _onSlotSwitched(detail) {
        const next = detail?.slot;
        if (next !== 'A' && next !== 'B') return;
        this.currentSlot = next;
        this._render();
        // Reapply z-order for the freshly-active slot's layers so they stack
        // correctly after MapLayers re-attached them to the map.
        setTimeout(() => this._applyZOrder(), 100);
        this._updateCompareButtons();
        this._notifyMaskRegistry();
    }

    /**
     * Tell the mask compositor (and any other listener) that the active
     * registry — entries, ordering, visibility, or mask roles — has changed.
     * The compositor recomputes the composite mask and updates the on-map
     * overlay. Coalesces with microtask scheduling on the consumer side.
     */
    _notifyMaskRegistry() {
        try {
            window.dispatchEvent(new CustomEvent('mask:registry-changed'));
        } catch (e) { /* no-op */ }
    }

    _toggleCollapse() {
        this.collapsed = !this.collapsed;
        if (this.listEl) {
            this.listEl.style.display = this.collapsed ? 'none' : '';
        }
        if (this.toggleBtn) {
            this.toggleBtn.textContent = this.collapsed ? '▶' : '▼';
        }
    }

    _render() {
        if (!this.sectionEl || !this.listEl) return;

        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];

        if (!registry || registry.length === 0) {
            this.sectionEl.style.display = 'none';
            return;
        }

        this.sectionEl.style.display = '';
        this.listEl.innerHTML = '';

        // Render top-to-bottom (reverse of z-order: last in array = top).
        for (let i = registry.length - 1; i >= 0; i--) {
            const row = registry[i];
            if (this._isGroup(row)) {
                this.listEl.appendChild(this._renderGroupRow(i, row));
            } else {
                this.listEl.appendChild(this._renderSingleRow(i, row.entry));
            }
        }
    }

    /**
     * Render a single (non-grouped) layer row. Mask chip is shown only for
     * binary analysis layers and is non-interactive — it's a marker, not a
     * cycling control. Mask composition is reachable only by drag-merging
     * two binary rows into a group.
     */
    _renderSingleRow(rowIdx, entry) {
        const item = document.createElement('div');
        item.className = 'layer-item' + (entry.visible ? ' layer-visible' : ' layer-hidden');
        item.dataset.rowIndex = rowIdx;
        item.dataset.kind = 'single';
        item.draggable = true;

        const typeIcon = this._getTypeIcon(entry.type);
        const safeName = this._escapeHtml(entry.name);
        // Marker shows only on bona-fide binary masks (post-threshold,
        // SAM3, segmentation, change-detection, binary-mask COG). Continuous
        // overlays (raw NDVI/MVI/score maps) get nothing.
        const showChip = (entry.type === 'analysis' && entry.isBinary === true);
        const maskChipHtml = showChip ? this._renderMaskChip() : '';

        item.innerHTML = `
            <span class="layer-drag-handle" title="Drag the row to reorder, drop on another mask to merge">⠿</span>
            <span class="layer-type-icon">${typeIcon}</span>
            <span class="layer-name" title="Double-click to rename">${safeName}</span>
            ${maskChipHtml}
            <span class="layer-visibility-indicator">${entry.visible ? 'ON' : 'OFF'}</span>
        `;

        item.addEventListener('click', (e) => {
            if (e.target.closest('.layer-name-edit')) return;
            if (e.target.closest('.layer-name')) return;
            if (e.target.closest('.layer-mask-chip')) return;
            this._toggleSingleVisibility(rowIdx);
        });

        // Disabled chip: don't let it propagate, but no cycling action.
        const chipEl = item.querySelector('.layer-mask-chip');
        if (chipEl) {
            chipEl.addEventListener('click', (e) => e.stopPropagation());
            chipEl.addEventListener('mousedown', (e) => e.stopPropagation());
            chipEl.draggable = false;
        }

        const nameEl = item.querySelector('.layer-name');
        nameEl.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            this._beginRename(item, entry);
        });

        item.addEventListener('dragstart', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.isContentEditable) {
                e.preventDefault();
                return;
            }
            if (e.target.closest && e.target.closest('.layer-mask-chip')) {
                e.preventDefault();
                return;
            }
            this._onDragStart(e, { kind: 'single', rowIdx });
        });
        item.addEventListener('dragover', (e) => this._onDragOver(e));
        item.addEventListener('dragenter', (e) => this._onDragEnter(e));
        item.addEventListener('dragleave', (e) => this._onDragLeave(e));
        item.addEventListener('drop', (e) => this._onDrop(e));
        item.addEventListener('dragend', (e) => this._onDragEnd(e));

        return item;
    }

    /**
     * Render a mask group row: items laid out left-to-right with operator
     * buttons interleaved. The whole row participates in reorder dragging
     * (handle on the far left) and individual items can be dragged out.
     */
    _renderGroupRow(rowIdx, row) {
        const groupVisible = row.visible !== false;
        const container = document.createElement('div');
        container.className = 'layer-item layer-group-row' + (groupVisible ? ' layer-visible' : ' layer-hidden');
        container.dataset.rowIndex = rowIdx;
        container.dataset.kind = 'group';
        container.dataset.groupId = row.id;
        container.draggable = true;

        // Drag handle (whole-group reorder)
        const handle = document.createElement('span');
        handle.className = 'layer-drag-handle layer-group-handle';
        handle.title = 'Drag to reorder this group';
        handle.textContent = '⠿';
        container.appendChild(handle);

        // Render items + operators interleaved.
        for (let k = 0; k < row.items.length; k++) {
            if (k > 0) {
                const op = document.createElement('span');
                op.className = 'layer-group-operator';
                op.dataset.opIndex = k - 1;
                const role = row.operators[k - 1] || 'inc';
                op.dataset.role = role;
                op.textContent = ({ inc: '∩', add: '∪', exc: '−' })[role] || '∩';
                op.title = 'Click to cycle: ∩ → ∪ → − → ∩';
                op.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this._cycleGroupOperator(rowIdx, k - 1);
                });
                op.addEventListener('mousedown', (e) => e.stopPropagation());
                op.draggable = false;
                container.appendChild(op);
            }

            const entry = row.items[k];
            const itemEl = document.createElement('div');
            itemEl.className = 'layer-group-item';
            itemEl.dataset.itemIndex = k;
            itemEl.draggable = true;

            const safeName = this._escapeHtml(entry.name);
            itemEl.innerHTML = `
                <span class="layer-type-icon">${this._getTypeIcon(entry.type)}</span>
                <span class="layer-group-name" title="${safeName} — double-click to rename">${safeName}</span>
                <span class="layer-group-remove" title="Remove from group">✕</span>
            `;

            // ✕ button: detach this item back to a single row.
            const removeBtn = itemEl.querySelector('.layer-group-remove');
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this._removeGroupItem(rowIdx, k);
            });
            removeBtn.addEventListener('mousedown', (e) => e.stopPropagation());

            // Double-click name → rename.
            const nameEl = itemEl.querySelector('.layer-group-name');
            nameEl.addEventListener('dblclick', (e) => {
                e.stopPropagation();
                this._beginRenameGroupItem(itemEl, entry);
            });

            // Per-item drag — to merge into other groups, drag out to
            // detach, OR drag onto another item in the *same* group to
            // swap (reorder within the group). Source identity carries
            // (rowIdx, itemIdx).
            itemEl.addEventListener('dragstart', (e) => {
                if (e.target.tagName === 'INPUT' || e.target.isContentEditable) {
                    e.preventDefault(); return;
                }
                if (e.target.closest('.layer-group-remove')) {
                    e.preventDefault(); return;
                }
                e.stopPropagation();
                this._onDragStart(e, { kind: 'groupItem', rowIdx, itemIdx: k });
            });
            itemEl.addEventListener('dragend', (e) => this._onDragEnd(e));

            // Within-group reorder: hovering another item in the SAME group
            // marks it as a swap target. Stops propagation so the row-level
            // merge/reorder logic does NOT fire for this case.
            itemEl.addEventListener('dragover', (e) => {
                if (this._dragSrc?.kind !== 'groupItem') return;
                if (this._dragSrc.rowIdx !== rowIdx) return;
                if (this._dragSrc.itemIdx === k) return;
                e.preventDefault();
                e.stopPropagation();
                if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
                this._dragTargetMode = 'groupReorder';
                this._dragGroupReorderTarget = { rowIdx, itemIdx: k };
                this._removeDropIndicator();
                this._clearMergeTargetClasses();
                container.querySelectorAll('.layer-group-item').forEach(el =>
                    el.classList.remove('group-item-drop-target'));
                itemEl.classList.add('group-item-drop-target');
            });
            itemEl.addEventListener('drop', (e) => {
                if (this._dragSrc?.kind !== 'groupItem') return;
                if (this._dragSrc.rowIdx !== rowIdx) return;
                if (this._dragSrc.itemIdx === k) return;
                e.preventDefault();
                e.stopPropagation();
                this._performGroupItemSwap(rowIdx, this._dragSrc.itemIdx, k);
            });

            container.appendChild(itemEl);
        }

        // Group ON/OFF indicator on the right — same affordance as single
        // rows. Click toggles whether the composite mask is rendered on the
        // map. The merged group behaves as one unified mask from here.
        const visIndicator = document.createElement('span');
        visIndicator.className = 'layer-visibility-indicator layer-group-visibility';
        visIndicator.textContent = groupVisible ? 'ON' : 'OFF';
        visIndicator.title = 'Click to toggle this combined mask on/off';
        visIndicator.addEventListener('click', (e) => {
            e.stopPropagation();
            this._toggleGroupVisibility(rowIdx);
        });
        visIndicator.addEventListener('mousedown', (e) => e.stopPropagation());
        container.appendChild(visIndicator);

        // Click anywhere inside the row toggles, except for elements that
        // already handle their own click (✕ remove, operator cycle, drag
        // handle, visibility indicator). Clicking item body / item name
        // single-click toggles too — name dblclick still triggers rename.
        container.addEventListener('click', (e) => {
            if (e.target.closest('.layer-group-remove')) return;
            if (e.target.closest('.layer-group-operator')) return;
            if (e.target.closest('.layer-group-handle')) return;
            if (e.target.closest('.layer-group-visibility')) return;
            if (e.target.closest('.layer-name-edit')) return; // active rename input
            this._toggleGroupVisibility(rowIdx);
        });

        // Whole-group drag (initiated from handle / row background, NOT from
        // child items — those have their own dragstart that stopPropagation).
        container.addEventListener('dragstart', (e) => {
            if (e.target.closest('.layer-group-item')) return;
            if (e.target.closest('.layer-group-operator')) {
                e.preventDefault(); return;
            }
            this._onDragStart(e, { kind: 'group', rowIdx });
        });
        container.addEventListener('dragover', (e) => this._onDragOver(e));
        container.addEventListener('drop', (e) => this._onDrop(e));

        return container;
    }

    _escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => (
            {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
        ));
    }

    /**
     * Render the binary-mask marker chip — a non-interactive badge that
     * indicates the layer is a binary mask and therefore eligible for
     * mask-group composition (drag this row onto another mask row to merge).
     */
    _renderMaskChip() {
        const ttl = 'Binary mask — drag onto another mask to combine (∩/∪/−)';
        return `<span class="layer-mask-chip" data-disabled="true" title="${ttl}">MASK</span>`;
    }

    _getTypeIcon(type) {
        switch (type) {
            case 'analysis': return '📊';
            case 'tile': return '🗺️';
            case 'image': return '🛰️';
            case 'processed': return '🖼️';
            case 'change-detection': return '🔀';
            default: return '📄';
        }
    }

    // ========== Rename ==========

    _beginRename(itemEl, entry) {
        const nameEl = itemEl.querySelector('.layer-name');
        if (!nameEl || nameEl.dataset.editing === '1') return;
        nameEl.dataset.editing = '1';

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'layer-name-edit';
        input.value = entry.name;
        input.setAttribute('aria-label', 'Rename layer');

        // Inline styles so we don't require CSS changes — readable against
        // the row's blue/grey background. Font matches surrounding text.
        input.style.cssText = `
            flex: 1; min-width: 0;
            padding: 2px 6px;
            font-size: 12px; font-weight: 500;
            border: 1px solid #1a73e8; border-radius: 4px;
            background: #fff; color: #333;
            outline: none;
        `;

        nameEl.replaceWith(input);
        input.focus();
        input.select();

        const commit = (save) => {
            if (!input.parentNode) return;  // already removed
            const newName = save ? input.value.trim() : entry.name;
            if (save && newName && newName !== entry.name) {
                entry.name = newName;
                entry.userRenamed = true;
                // Propagate into PlatformController.analyses so the Change
                // Detection picker shows the new label too.
                this._propagateRename(entry.id, newName);
            }
            // Re-render: easiest way to rebuild the row cleanly.
            this._render();
        };

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); commit(true); }
            else if (e.key === 'Escape') { e.preventDefault(); commit(false); }
        });
        input.addEventListener('blur', () => commit(true));
        // Prevent the item-level click from firing through the input.
        input.addEventListener('click', e => e.stopPropagation());
        input.addEventListener('dblclick', e => e.stopPropagation());
        // Also prevent dragstart while editing.
        input.addEventListener('mousedown', e => e.stopPropagation());
    }

    _propagateRename(layerId, newName) {
        const platform = window.platform;
        if (!platform?._slots) return;
        let hitSlot = null;
        for (const slotId of ['A', 'B']) {
            const analyses = platform._slots[slotId]?.analyses;
            if (!Array.isArray(analyses)) continue;
            const hit = analyses.find(a => a.id === layerId);
            if (hit) {
                hit.name = newName;
                hitSlot = slotId;
                if (typeof platform._updateSlotToolbar === 'function') {
                    try { platform._updateSlotToolbar(); } catch (_) { /* no-op */ }
                }
                break;
            }
        }
        // Broadcast so downstream consumers (Change Detection dropdowns,
        // existing CD result cards, compare-mode picker) can refresh their
        // cached display names without polling.
        window.dispatchEvent(new CustomEvent('slot-analysis:renamed', {
            detail: { layerId, newName, slotId: hitSlot },
        }));
    }

    _setEntryVisible(entry, visible) {
        if (!entry || !this.mapManager?.map) return;
        const map = this.mapManager.map;
        if (visible) {
            if (entry.leafletLayer && !map.hasLayer(entry.leafletLayer)) {
                map.addLayer(entry.leafletLayer);
            }
            entry.visible = true;
        } else {
            if (entry.leafletLayer && map.hasLayer(entry.leafletLayer)) {
                map.removeLayer(entry.leafletLayer);
            }
            entry.visible = false;
        }
    }

    /**
     * Toggle a leaflet layer's map presence without touching `entry.visible`.
     * Used to hide individual mask overlays while they live inside a group
     * (the composite overlay represents them on the map) and put them back
     * on the map when detached. The user-intent flag stays true throughout.
     */
    _setEntryOnMap(entry, on) {
        if (!entry || !this.mapManager?.map) return;
        const map = this.mapManager.map;
        const layer = entry.leafletLayer;
        if (!layer) return;
        if (on) {
            if (!map.hasLayer(layer)) map.addLayer(layer);
        } else {
            if (map.hasLayer(layer)) map.removeLayer(layer);
        }
    }

    _toggleSingleVisibility(rowIdx) {
        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];
        const row = registry[rowIdx];
        if (!this._isSingle(row)) return;
        this._setEntryVisible(row.entry, !row.entry.visible);
        if (row.entry.visible) setTimeout(() => this._applyZOrder(), 100);
        this._render();
        this._notifyMaskRegistry();
    }

    _toggleGroupVisibility(rowIdx) {
        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];
        const row = registry[rowIdx];
        if (!this._isGroup(row)) return;
        // Group visibility controls whether the composite overlay is shown.
        // Individual item layers stay off the map either way (the composite
        // represents them); when the user pulls an item out, _setEntryOnMap
        // brings it back.
        row.visible = (row.visible === false);
        this._render();
        this._notifyMaskRegistry();
    }

    /**
     * Cycle the operator at (rowIdx, opIdx) in a group: inc → add → exc → inc.
     */
    _cycleGroupOperator(rowIdx, opIdx) {
        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];
        const row = registry[rowIdx];
        if (!this._isGroup(row)) return;
        if (opIdx < 0 || opIdx >= row.operators.length) return;
        const order = ['inc', 'add', 'exc'];
        const cur = row.operators[opIdx];
        const i = order.indexOf(cur);
        row.operators[opIdx] = order[(i + 1) % order.length];
        this._render();
        this._notifyMaskRegistry();
    }

    /**
     * ✕ button: pull an item out of a group and re-insert it as a single row
     * positioned just *above* (top-of-list direction = end of registry array)
     * the remaining group, so the user can see it land somewhere predictable.
     */
    _removeGroupItem(rowIdx, itemIdx) {
        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];
        const row = registry[rowIdx];
        if (!this._isGroup(row)) return;
        const removed = this._detachFromGroup(registry, rowIdx, itemIdx);
        if (!removed) return;
        // Detached → individual layer goes back on the map (if user intent
        // was visible). If the group dissolved (1 item left), the surviving
        // item also returns to the map.
        this._setEntryOnMap(removed, removed.visible !== false);
        const dissolvedRow = registry[rowIdx];
        if (this._isSingle(dissolvedRow)) {
            this._setEntryOnMap(dissolvedRow.entry, dissolvedRow.entry.visible !== false);
        }
        // Re-insert as a single just *above* whatever the group became.
        const insertAt = Math.min(registry.length, rowIdx + 1);
        registry.splice(insertAt, 0, this._wrapSingle(removed));
        this._render();
        setTimeout(() => this._applyZOrder(), 50);
        this._notifyMaskRegistry();
    }

    /**
     * Swap two items within the same group. Operators stay in their slot
     * positions, so swapping items effectively rewires which operator joins
     * which pair — letting the user freely re-order the operation chain.
     */
    _performGroupItemSwap(rowIdx, srcIdx, dstIdx) {
        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];
        const row = registry[rowIdx];
        if (!this._isGroup(row)) return;
        if (srcIdx === dstIdx) return;
        if (srcIdx < 0 || srcIdx >= row.items.length) return;
        if (dstIdx < 0 || dstIdx >= row.items.length) return;
        const tmp = row.items[srcIdx];
        row.items[srcIdx] = row.items[dstIdx];
        row.items[dstIdx] = tmp;
        // Reset drag state so dragend doesn't try to undo anything.
        this._dragSrc = null;
        this._dragSrcIndex = null;
        this._dragTargetMode = null;
        this._dragGroupReorderTarget = null;
        this._dragMergeTargetRowIdx = -1;
        this._dragDropDisplayIndex = null;
        this._removeDropIndicator();
        this._clearMergeTargetClasses();
        if (this.listEl) {
            this.listEl.querySelectorAll('.group-item-drop-target').forEach(el =>
                el.classList.remove('group-item-drop-target'));
        }
        this._render();
        this._notifyMaskRegistry();
    }

    _beginRenameGroupItem(itemEl, entry) {
        const nameEl = itemEl.querySelector('.layer-group-name');
        if (!nameEl || nameEl.dataset.editing === '1') return;
        nameEl.dataset.editing = '1';

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'layer-name-edit';
        input.value = entry.name;
        input.style.cssText = `
            flex: 1; min-width: 0;
            padding: 1px 4px; font-size: 11px;
            border: 1px solid #1a73e8; border-radius: 3px;
            background: #fff; color: #333; outline: none;
        `;
        nameEl.replaceWith(input);
        input.focus(); input.select();

        const commit = (save) => {
            if (!input.parentNode) return;
            const newName = save ? input.value.trim() : entry.name;
            if (save && newName && newName !== entry.name) {
                entry.name = newName;
                entry.userRenamed = true;
                this._propagateRename(entry.id, newName);
            }
            this._render();
        };
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); commit(true); }
            else if (e.key === 'Escape') { e.preventDefault(); commit(false); }
        });
        input.addEventListener('blur', () => commit(true));
        input.addEventListener('click', e => e.stopPropagation());
        input.addEventListener('mousedown', e => e.stopPropagation());
    }

    /**
     * Apply registry order to the map.
     *
     * The hard problem: Sentinel-2 preview tiles live in `dataPane` (z=450)
     * while analysis overlays live in `analysisPane` (z=475). No amount of
     * `setZIndex` or DOM reorder *within* a pane can lift a dataPane tile
     * above an analysisPane overlay — the panes themselves stack, and the
     * tile's container is parented to dataPane.
     *
     * Fix: when a layer first appears in this panel's registry, **reparent
     * its container into a single shared pane** (`analysisPane`) so all
     * controllable layers share a stacking context. After that, within-pane
     * stacking via `setZIndex` + `appendChild` order works as expected.
     *
     * Reparenting moves the existing container DOM node — Leaflet keeps
     * painting tiles into it because tile positions are absolute relative
     * to the map's CRS, not the parent. The pane the layer was *configured*
     * to use stays in `options.pane` but is unused at runtime.
     */
    _applyZOrder() {
        const map = this.mapManager?.map;
        const targetPane = map && map.getPane && map.getPane('analysisPane');
        const registry = this._flattenRows(this._slotLayers[this.currentSlot][this._getActiveTab()]);

        for (let i = 0; i < registry.length; i++) {
            const entry = registry[i];
            if (!entry.visible || !entry.leafletLayer) continue;
            // Skip entries currently hidden because they're inside a group
            // (the composite overlay represents them).
            if (map && !map.hasLayer(entry.leafletLayer)) continue;
            const layer = entry.leafletLayer;
            const zVal = 451 + i; // base z within the shared pane

            // (1) setZIndex if the layer exposes it.
            try {
                if (typeof layer.setZIndex === 'function') layer.setZIndex(zVal);
            } catch (e) { /* fall through */ }

            // (2) Reparent the container into the shared pane (one-time per
            // layer per slot is enough; subsequent calls are no-ops because
            // the container is already in `targetPane`).
            const container =
                (typeof layer.getContainer === 'function' && layer.getContainer()) ||
                (typeof layer.getElement === 'function' && layer.getElement()) ||
                layer._container || null;
            if (container && targetPane && container.parentNode !== targetPane) {
                try { targetPane.appendChild(container); } catch (e) { /* no-op */ }
            }

            // (3) Within the shared pane, registry-bottom→top maps to
            // DOM-first→last via appendChild. Last registered = on top.
            if (container && container.parentNode) {
                try { container.parentNode.appendChild(container); } catch (e) { /* no-op */ }
                try { container.style.zIndex = String(zVal); } catch (e) { /* no-op */ }
            }
        }
    }

    // ========== Drag and drop ==========

    // Drag source descriptor is one of:
    //   { kind: 'single',    rowIdx }
    //   { kind: 'group',     rowIdx }
    //   { kind: 'groupItem', rowIdx, itemIdx }
    // Cursor position within a hovered row chooses intent:
    //   - top 30% / bottom 30%  → reorder (above / below this row)
    //   - middle 40%            → merge (combine into this row's mask group)
    _onDragStart(e, descriptor) {
        this._dragSrc = descriptor;
        this._dragSrcIndex = (descriptor && typeof descriptor.rowIdx === 'number')
            ? descriptor.rowIdx : null;
        if (window.__maskDebug) console.log('[mask-merge] dragstart', descriptor);
        const item = e.currentTarget?.closest?.('.layer-item') || e.target.closest('.layer-item');
        if (item) item.classList.add('dragging');
        if (e.dataTransfer) {
            e.dataTransfer.effectAllowed = 'move';
            try { e.dataTransfer.setData('text/plain', String(this._dragSrcIndex ?? '')); } catch (_) { /* IE quirk */ }
        }
    }

    _onDragOver(e) {
        e.preventDefault();
        if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
    }

    _onDragEnter(e) { /* no-op */ }
    _onDragLeave(e) { /* no-op */ }

    _isBinaryRow(row) {
        if (!row) return false;
        if (this._isGroup(row)) return true;
        return row.entry?.isBinary === true && row.entry?.type === 'analysis';
    }

    _dragSourceIsBinary() {
        if (!this._dragSrc) return false;
        const reg = this._slotLayers[this.currentSlot][this._getActiveTab()];
        if (this._dragSrc.kind === 'groupItem') {
            const row = reg[this._dragSrc.rowIdx];
            if (!this._isGroup(row)) return false;
            return row.items?.[this._dragSrc.itemIdx]?.isBinary === true;
        }
        return this._isBinaryRow(reg[this._dragSrc.rowIdx]);
    }

    _clearMergeTargetClasses() {
        if (!this.listEl) return;
        this.listEl.querySelectorAll('.merge-drop-target').forEach(el =>
            el.classList.remove('merge-drop-target'));
    }

    /**
     * List-level dragover: compute the drop position based on cursor Y vs.
     * each item's vertical midpoint. Cursor above the first item → drop at
     * top (display position 0); below the last → drop at bottom. Render a
     * horizontal indicator at the resolved boundary so the user sees exactly
     * where the layer will land.
     *
     * Display indices are top-down (0 = top), registry indices are bottom-up
     * (length-1 = top). The conversion is `registry = items.length - 1 - display`.
     */
    _onListDragOver(e) {
        e.preventDefault();
        if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
        if (!this._dragSrc) {
            if (window.__maskDebug) console.warn('[mask-merge] no _dragSrc');
            return;
        }

        const items = Array.from(this.listEl.querySelectorAll('.layer-item'));
        if (items.length === 0) return;

        const y = e.clientY;

        // Find a row currently under the cursor (if any).
        let hoveredEl = null;
        for (let i = 0; i < items.length; i++) {
            const r = items[i].getBoundingClientRect();
            if (y >= r.top && y <= r.bottom) { hoveredEl = items[i]; break; }
        }

        // Default reorder drop index from item midpoints.
        let displayDropIndex = items.length;
        for (let i = 0; i < items.length; i++) {
            const r = items[i].getBoundingClientRect();
            if (y < r.top + r.height / 2) { displayDropIndex = i; break; }
        }

        let mergeTargetRowIdx = -1;
        const srcIsBinary = this._dragSourceIsBinary();
        if (window.__maskDebug) {
            console.log('[mask-merge] dragover', {
                srcKind: this._dragSrc?.kind,
                srcRowIdx: this._dragSrc?.rowIdx,
                srcIsBinary,
                hoveredRowIdx: hoveredEl ? parseInt(hoveredEl.dataset.rowIndex, 10) : null,
            });
        }
        if (hoveredEl && srcIsBinary) {
            const r = hoveredEl.getBoundingClientRect();
            const relY = (y - r.top) / Math.max(1, r.height);
            if (relY > 0.3 && relY < 0.7) {
                const targetRowIdx = parseInt(hoveredEl.dataset.rowIndex, 10);
                const reg = this._slotLayers[this.currentSlot][this._getActiveTab()];
                const targetRow = reg[targetRowIdx];
                const isSelfRow = (this._dragSrc.rowIdx === targetRowIdx);
                const wholeRowSrc = (this._dragSrc.kind !== 'groupItem');
                const groupItemOnOwnRow = (this._dragSrc.kind === 'groupItem' && isSelfRow);
                const targetBinary = this._isBinaryRow(targetRow);
                if (window.__maskDebug) {
                    console.log('[mask-merge] middle-zone', {
                        targetRowIdx, targetBinary, isSelfRow, wholeRowSrc, groupItemOnOwnRow,
                    });
                }
                if (targetBinary && !(wholeRowSrc && isSelfRow) && !groupItemOnOwnRow) {
                    mergeTargetRowIdx = targetRowIdx;
                }
            }
        }

        if (mergeTargetRowIdx >= 0) {
            this._dragTargetMode = 'merge';
            this._dragMergeTargetRowIdx = mergeTargetRowIdx;
            this._dragDropDisplayIndex = null;
            this._removeDropIndicator();
            this._clearMergeTargetClasses();
            for (const el of items) {
                if (parseInt(el.dataset.rowIndex, 10) === mergeTargetRowIdx) {
                    el.classList.add('merge-drop-target');
                }
            }
        } else {
            this._dragTargetMode = 'reorder';
            this._dragMergeTargetRowIdx = -1;
            this._dragDropDisplayIndex = displayDropIndex;
            this._clearMergeTargetClasses();
            this._renderDropIndicator(displayDropIndex, items);
        }
    }

    _onListDragLeave(e) {
        // Don't hide the indicator on dragleave — the document-level handler
        // takes over outside the panel and keeps the indicator pinned to the
        // top or bottom edge. Final cleanup happens in `_onDragEnd`.
    }

    /**
     * Document-level fallback. Fires while the cursor is anywhere on the
     * page during a layer-row drag. If cursor is above the panel → snap to
     * top; below → snap to bottom. Lets the user "swipe past" the panel
     * boundary in either direction without precise aim.
     */
    _onDocDragOver(e) {
        if (!this._dragSrc) return;
        if (!this.sectionEl) return;
        const rect = this.sectionEl.getBoundingClientRect();
        if (e.clientY >= rect.top && e.clientY <= rect.bottom &&
            e.clientX >= rect.left && e.clientX <= rect.right) {
            return;
        }
        e.preventDefault();
        if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';

        const items = Array.from(this.listEl?.querySelectorAll('.layer-item') || []);
        if (items.length === 0) return;

        // Outside the panel: only reorder makes sense, never merge.
        const snapTop = e.clientY < rect.top + rect.height / 2;
        this._dragTargetMode = 'reorder';
        this._dragMergeTargetRowIdx = -1;
        this._dragDropDisplayIndex = snapTop ? 0 : items.length;
        this._clearMergeTargetClasses();
        this._renderDropIndicator(this._dragDropDisplayIndex, items);
    }

    _onDocDrop(e) {
        if (!this._dragSrc) return;
        if (!this.sectionEl) return;
        const rect = this.sectionEl.getBoundingClientRect();
        if (e.clientY >= rect.top && e.clientY <= rect.bottom &&
            e.clientX >= rect.left && e.clientX <= rect.right) return;
        this._onListDrop(e);
    }

    _onListDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        if (window.__maskDebug) console.log('[mask-merge] drop', {
            mode: this._dragTargetMode,
            mergeTargetRowIdx: this._dragMergeTargetRowIdx,
            displayDropIndex: this._dragDropDisplayIndex,
        });
        if (!this._dragSrc) {
            this._removeDropIndicator();
            this._clearMergeTargetClasses();
            return;
        }

        // Within-group swap drop is handled by the per-item drop handler;
        // if it didn't fire (cursor moved off the target item before release),
        // treat the drop as a no-op rather than fall through to merge/reorder.
        if (this._dragTargetMode === 'groupReorder') {
            this._dragMergeTargetRowIdx = -1;
            this._dragTargetMode = null;
            this._dragDropDisplayIndex = null;
            this._dragGroupReorderTarget = null;
            this._removeDropIndicator();
            this._clearMergeTargetClasses();
            if (this.listEl) {
                this.listEl.querySelectorAll('.group-item-drop-target').forEach(el =>
                    el.classList.remove('group-item-drop-target'));
            }
            return;
        }

        if (this._dragTargetMode === 'merge' && this._dragMergeTargetRowIdx >= 0) {
            this._performMerge(this._dragMergeTargetRowIdx);
        } else if (this._dragDropDisplayIndex != null) {
            this._performReorder(this._dragDropDisplayIndex);
        }

        this._dragMergeTargetRowIdx = -1;
        this._dragTargetMode = null;
        this._dragDropDisplayIndex = null;
        this._removeDropIndicator();
        this._clearMergeTargetClasses();
        this._render();
        setTimeout(() => this._applyZOrder(), 50);
        this._notifyMaskRegistry();
    }

    _performReorder(targetDisplayIndex) {
        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];
        const total = registry.length;
        const targetDisplay = Math.max(0, Math.min(total, targetDisplayIndex));
        const targetRegistryRaw = total - targetDisplay;

        if (this._dragSrc.kind === 'groupItem') {
            // Drag-out-of-group: detach + insert as a single row.
            const { rowIdx, itemIdx } = this._dragSrc;
            const removed = this._detachFromGroup(registry, rowIdx, itemIdx);
            if (!removed) return;
            // Detached layer returns to the map (user intent stays).
            this._setEntryOnMap(removed, removed.visible !== false);
            // If dissolve made the group → single, that survivor also returns.
            const dissolvedRow = registry[rowIdx];
            if (this._isSingle(dissolvedRow)) {
                this._setEntryOnMap(dissolvedRow.entry, dissolvedRow.entry.visible !== false);
            }
            const adjusted = Math.max(0, Math.min(registry.length, targetRegistryRaw));
            registry.splice(adjusted, 0, this._wrapSingle(removed));
            return;
        }

        // Whole-row reorder (single OR group).
        const srcRegistryIndex = this._dragSrc.rowIdx;
        if (srcRegistryIndex < 0 || srcRegistryIndex >= registry.length) return;
        const [moved] = registry.splice(srcRegistryIndex, 1);
        const adjusted = (srcRegistryIndex < targetRegistryRaw)
            ? targetRegistryRaw - 1 : targetRegistryRaw;
        const finalIdx = Math.max(0, Math.min(registry.length, adjusted));
        registry.splice(finalIdx, 0, moved);
    }

    /**
     * Merge the drag source into the row at `targetRowIdx`. Default operator
     * inserted between newly adjacent items is 'inc' (∩).
     *
     *   single + single   → new group of 2
     *   single + group    → group gains one tail item
     *   group + group     → first absorbs the second's items + operators
     *   groupItem + any   → that item is detached from its source group and
     *                       merged into the target as a single
     */
    _performMerge(targetRowIdx) {
        const registry = this._slotLayers[this.currentSlot][this._getActiveTab()];
        const targetRow = registry[targetRowIdx];
        if (!targetRow) return;
        const src = this._dragSrc;
        if (!src) return;

        let srcEntries = [];
        let srcOperators = [];
        let srcRemovalKind = null; // 'item' | 'row'

        if (src.kind === 'groupItem') {
            const srcRow = registry[src.rowIdx];
            if (!this._isGroup(srcRow)) return;
            const item = srcRow.items[src.itemIdx];
            if (!item || item.isBinary !== true) return;
            srcEntries = [item];
            srcRemovalKind = 'item';
        } else if (src.kind === 'single') {
            const srcRow = registry[src.rowIdx];
            if (!this._isSingle(srcRow)) return;
            if (srcRow.entry.isBinary !== true) return;
            if (src.rowIdx === targetRowIdx) return;
            srcEntries = [srcRow.entry];
            srcRemovalKind = 'row';
        } else if (src.kind === 'group') {
            const srcRow = registry[src.rowIdx];
            if (!this._isGroup(srcRow)) return;
            if (src.rowIdx === targetRowIdx) return;
            srcEntries = srcRow.items.slice();
            srcOperators = srcRow.operators.slice();
            srcRemovalKind = 'row';
        }
        if (srcEntries.length === 0) return;

        // Mutate target first (in place), then remove source — order matters
        // so targetRow's reference stays valid.
        if (this._isGroup(targetRow)) {
            targetRow.operators.push('inc', ...srcOperators);
            targetRow.items.push(...srcEntries);
        } else if (this._isSingle(targetRow)) {
            const targetEntry = targetRow.entry;
            registry[targetRowIdx] = {
                kind: 'group',
                id: this._newGroupId(),
                items: [targetEntry, ...srcEntries],
                operators: ['inc', ...srcOperators],
                visible: true,
            };
            // Target's individual layer is now represented by the composite —
            // pull it off the map. Its `visible` user-intent flag stays true.
            this._setEntryOnMap(targetEntry, false);
        }

        // Newly-merged items get hidden too — composite overlay stands in.
        for (const e of srcEntries) this._setEntryOnMap(e, false);

        if (srcRemovalKind === 'item') {
            this._detachFromGroup(registry, src.rowIdx, src.itemIdx);
        } else if (srcRemovalKind === 'row') {
            registry.splice(src.rowIdx, 1);
        }
    }

    _renderDropIndicator(displayIndex, items) {
        let bar = this.listEl.querySelector('.layer-drop-indicator');
        if (!bar) {
            bar = document.createElement('div');
            bar.className = 'layer-drop-indicator';
            this.listEl.appendChild(bar);
        }
        const listRect = this.listEl.getBoundingClientRect();
        let top;
        if (displayIndex >= items.length) {
            const last = items[items.length - 1].getBoundingClientRect();
            top = last.bottom - listRect.top + this.listEl.scrollTop;
        } else {
            const r = items[displayIndex].getBoundingClientRect();
            top = r.top - listRect.top + this.listEl.scrollTop;
        }
        bar.style.top = `${top - 1}px`;
    }

    _removeDropIndicator() {
        const bar = this.listEl?.querySelector('.layer-drop-indicator');
        if (bar) bar.remove();
    }

    _onDrop(e) {
        // Per-row drop: defer to the list-level handler (authoritative).
        e.preventDefault();
    }

    _onDragEnd(e) {
        this._dragSrc = null;
        this._dragSrcIndex = null;
        this._dragDropDisplayIndex = null;
        this._dragTargetMode = null;
        this._dragMergeTargetRowIdx = -1;
        this._dragGroupReorderTarget = null;
        this._removeDropIndicator();
        this._clearMergeTargetClasses();
        if (this.listEl) {
            this.listEl.querySelectorAll('.layer-item').forEach(item => {
                item.classList.remove('dragging', 'drag-over');
            });
            this.listEl.querySelectorAll('.group-item-drop-target').forEach(el =>
                el.classList.remove('group-item-drop-target'));
        }
    }

    _updateCompareButtons() {
        // Compare mode can now surface layers from both slots AND
        // change-detection results, so count across all of them — not just
        // the active-slot registry.
        const crossSlot = this.getAllAnalysisLayersBySlot();
        const cdCount = window.platform?.changeDetection?.results?.length || 0;
        const count = crossSlot.A.length + crossSlot.B.length + cdCount;
        const compareBtn = document.getElementById('compare-btn');
        const localCompareBtn = document.getElementById('local-compare-btn');

        if (this.currentTab === 'search' && compareBtn) {
            compareBtn.disabled = count < 2;
        }
        if (this.currentTab === 'local-image' && localCompareBtn) {
            localCompareBtn.disabled = count < 2;
        }
    }

    /** Get all layers for the current tab, scoped to the active slot.
     *  Returns a flat entry[] (rows are flattened) for backward compatibility
     *  with consumers that pre-date the row model. */
    getCurrentLayers() {
        return this._flattenRows(this._slotLayers[this.currentSlot][this._getActiveTab()] || []);
    }

    /** Get visible layers for the current tab, scoped to the active slot. */
    getVisibleLayers() {
        return this.getCurrentLayers().filter(l => l.visible);
    }

    /** Get the row[] for the current tab. Used by the mask compositor to
     *  read group structure (items + operators). */
    getMaskRows() {
        return this._slotLayers[this.currentSlot][this._getActiveTab()] || [];
    }

    /** Console diagnostic: print the current registry shape — useful when
     *  the user can't tell why the binary-mask marker isn't appearing. */
    inspectMasks() {
        const rows = this.getMaskRows();
        const flat = rows.map((r, i) => r.kind === 'single'
            ? { idx: i, kind: 'single', name: r.entry.name, type: r.entry.type, isBinary: r.entry.isBinary, visible: r.entry.visible }
            : { idx: i, kind: 'group',  items: r.items.map(e => `${e.name}(bin=${e.isBinary})`).join(' | '), operators: r.operators.join(',') });
        console.table(flat);
        return flat;
    }

    /**
     * Flatten all analysis-tab registrations across both slots into a single
     * `{A: [...], B: [...]}` shape. Used by compare mode so the user can pick
     * layers from the non-active slot without having to switch slots.
     *
     * Stashed-slot entries still carry their original `leafletLayer` (they
     * were just detached from the map on slot switch), so they can be cloned
     * straight onto the secondary map.
     */
    getAllAnalysisLayersBySlot() {
        const out = { A: [], B: [] };
        for (const slot of ['A', 'B']) {
            const regs = this._slotLayers[slot] || {};
            for (const tab of Object.keys(regs)) {
                for (const entry of this._flattenRows(regs[tab] || [])) {
                    // Change-detection overlays also dispatch `layer:added`
                    // (they use mapManager.showAnalysisLayer), so skip them
                    // here to avoid duplicating with the CD section that
                    // ChangeDetectionController exposes directly.
                    if (entry.id && entry.id.startsWith('chg_')) continue;
                    out[slot].push(entry);
                }
            }
        }
        return out;
    }
}
