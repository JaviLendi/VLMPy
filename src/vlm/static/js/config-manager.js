/**
 * UnifiedConfigManager - Unified manager for airfoil and plane (wing) configurations.
 * Combines functionality from airfoil-config-manager.js and plane-config-manager.js.
 *
 * Author: Javier Lendínez
 * SPDX-License-Identifier: AGPL-3.0-only
 */
class UnifiedConfigManager {
    constructor() {
        this.apiBase = '/api';
        this.cache = new Map();
        this.controllers = new Map(); // AbortController per endpoint.
        this.currentType = null; // 'airfoil' | 'plane' (set when opening a modal).
        // DOM elements (look for airfoil-form or wing-form).
        this.elements = {
            messageContainer: document.getElementById('message-container'),
            plotArea: document.getElementById('plot-area'),
            airfoilForm: document.getElementById('airfoil-form') || null,
            wingForm: document.getElementById('wing-form') || null
        };

        this.config = { messageTimeout: 5000, debug: false };

        // Precompiled templates that return strings.
        this.templates = {
            modalHTML: (title, body, footer = '', headerExtra = '') => `
                <div class="modal-overlay modern-modal" data-modal="true" aria-hidden="false">
                    <div class="modal-content modern">
                        <div class="modal-header">
                            <div style="display:flex;flex-direction:column;gap:0.5rem;width:100%">
                                <div style="display:flex;align-items:center;justify-content:space-between;">
                                    <h3 class="modal-title">${title}</h3>
                                    <button type="button" class="modal-close" data-action="close" aria-label="Close"><i class="fa fa-xmark" aria-hidden="true"></i></button>
                                </div>
                                ${headerExtra}
                            </div>
                        </div>
                        <div class="modal-body">${body}</div>
                        ${footer ? `<div class="modal-footer">${footer}</div>` : ''}
                    </div>
                </div>
            `,
            saveForm: (type) => {
                const pattern = "^[a-zA-Z0-9_\\-]+$";
                if (type === 'airfoil') {
                    return `
                        <form id="save-config-form" data-form="save-config" class="form">
                            <div class="form-group">
                                <label for="config-name">Configuration Name:</label>
                                <input type="text" id="config-name" name="name" class="input" placeholder="e.g., My airfoil config" maxlength="80" required>
                            </div>
                            <div class="form-group">
                                <label for="config-filename">Filename:</label>
                                <input type="text" id="config-filename" name="filename" class="input" placeholder="e.g., my_airfoil_config" pattern="^[a-zA-Z0-9_-]+$" maxlength="50" required title="Only letters, numbers, - and _ allowed">
                            </div>
                            <div class="form-group">
                                <label for="config-description">Description (optional):</label>
                                <textarea id="config-description" name="description" rows="3" maxlength="200" class="textarea" placeholder="Brief description..."></textarea>
                            </div>
                        </form>
                    `;
                }
                return `
                    <form id="save-config-form" data-form="save-config" class="form">
                        <div class="form-group">
                            <label for="config-name">Configuration Name:</label>
                            <input type="text" id="config-name" name="name" required placeholder="e.g., My wing configuration">
                        </div>
                        <div class="form-group">
                            <label for="config-filename">Filename:</label>
                            <input type="text" id="config-filename" name="filename" required placeholder="e.g., my_configuration" pattern="^[a-zA-Z0-9_-]+$" title="Only letters, numbers, - and _ allowed">
                        </div>
                        <div class="form-group">
                            <label for="config-description">Description (optional):</label>
                            <textarea id="config-description" name="description" rows="3" placeholder="Describe main characteristics..."></textarea>
                        </div>
                    </form>
                `;
            },
            airfoilThumbnailSVG: (cfg) => {
                const code = String(cfg.naca || '0012').replace(/\D/g, '').padStart(4, '0').slice(-4);
                const camber = Number(code[0]) / 100;
                const position = Number(code[1]) / 10;
                const thickness = Math.min(Math.max(Number(code.slice(2)) / 100, 0.02), 0.3);
                const upper = [];
                const lower = [];
                const camberLine = [];
                const samples = 32;
                for (let index = 0; index <= samples; index += 1) {
                    const fraction = (1 - Math.cos(Math.PI * index / samples)) / 2;
                    const thicknessLine = 5 * thickness * (
                        0.2969 * Math.sqrt(fraction) - 0.1260 * fraction
                        - 0.3516 * fraction ** 2 + 0.2843 * fraction ** 3
                        - 0.1036 * fraction ** 4
                    );
                    let camberHeight = 0;
                    let camberSlope = 0;
                    if (camber && position && fraction <= position) {
                        camberHeight = camber / position ** 2 * (2 * position * fraction - fraction ** 2);
                        camberSlope = 2 * camber / position ** 2 * (position - fraction);
                    } else if (camber && position) {
                        camberHeight = camber / (1 - position) ** 2 * (1 - 2 * position + 2 * position * fraction - fraction ** 2);
                        camberSlope = 2 * camber / (1 - position) ** 2 * (position - fraction);
                    }
                    const angle = Math.atan(camberSlope);
                    const upperX = 16 + (fraction - thicknessLine * Math.sin(angle)) * 148;
                    const lowerX = 16 + (fraction + thicknessLine * Math.sin(angle)) * 148;
                    upper.push(`${upperX.toFixed(2)},${(36 - (camberHeight + thicknessLine * Math.cos(angle)) * 210).toFixed(2)}`);
                    lower.unshift(`${lowerX.toFixed(2)},${(36 - (camberHeight - thicknessLine * Math.cos(angle)) * 210).toFixed(2)}`);
                    camberLine.push(`${(16 + fraction * 148).toFixed(2)},${(36 - camberHeight * 210).toFixed(2)}`);
                }
                const outline = upper.concat(lower).join(' ');
                return `<svg class="config-thumbnail-svg airfoil-thumbnail-svg" viewBox="0 0 180 72" role="img" aria-label="NACA ${code} airfoil profile" focusable="false">
                    <path class="config-thumbnail-grid" d="M12 18H168 M12 36H168 M12 54H168" />
                    <polygon class="airfoil-thumbnail-shape" points="${outline}" />
                    <polyline class="airfoil-thumbnail-camber" points="${camberLine.join(' ')}" />
                </svg>`;
            },
            wingThumbnailSVG: (cfg) => {
                const inputSections = cfg.wing_sections_preview || cfg.wing_sections || cfg.wingsections || [];
                const horizontal = cfg.horizontal_stabilizer_preview || cfg.horizontal_stabilizer || cfg.horizontalstabilizer;
                const vertical = cfg.vertical_stabilizer_preview || cfg.vertical_stabilizer || cfg.verticalstabilizer;
                const polygons = [];
                const addSurface = (surfaceSections, type, symmetric) => {
                    let leading = [0, 0, 0];
                    let previousChord = null;
                    surfaceSections.forEach(section => {
                        if (!section || typeof section !== 'object') return;
                        const rootChord = Number(section.chord_root ?? section.chordroot);
                        const tipChord = Number(section.chord_tip ?? section.chordtip);
                        const span = Number(section.span_fraction ?? section.spanfraction);
                        const sweep = Number(section.sweep ?? 0);
                        const dihedral = Number(section.dihedral ?? (symmetric ? 0 : Math.PI / 2));
                        const xOffset = Number(section.x_translate ?? 0);
                        const zOffset = Number(section.z_translate ?? 0);
                        if (![rootChord, tipChord, span, sweep, dihedral, xOffset, zOffset].every(Number.isFinite) || rootChord <= 0 || tipChord <= 0 || span <= 0) return;
                        const length = symmetric ? span / 2 : span;
                        const next = [
                            leading[0] + Math.tan(sweep) * length * (symmetric ? Math.cos(dihedral) : 1),
                            leading[1] + length * Math.cos(dihedral),
                            leading[2] + length * Math.sin(dihedral),
                        ];
                        const chord = previousChord ?? rootChord;
                        const vertices = [
                            [leading[0], leading[1], leading[2]],
                            [next[0], next[1], next[2]],
                            [next[0] + tipChord, next[1], next[2]],
                            [leading[0] + chord, leading[1], leading[2]],
                        ];
                        for (const side of symmetric ? [-1, 1] : [1]) {
                            polygons.push({type, points: vertices.map(([x, y, z]) => [x + xOffset, y * side, z + zOffset])});
                        }
                        leading = next;
                        previousChord = tipChord;
                    });
                };
                addSurface(Array.isArray(inputSections) ? inputSections : [], 'wing', true);
                if (horizontal) addSurface([horizontal], 'horizontal-stabilizer', true);
                if (vertical) addSurface([vertical], 'vertical-stabilizer', false);

                const project = ([x, y, z]) => [y + 0.35 * x, 0.55 * x - 0.12 * y - 0.85 * z];
                const projected = polygons.map(polygon => ({...polygon, points: polygon.points.map(project)}));
                const vertices = projected.flatMap(polygon => polygon.points);
                const minX = vertices.length ? Math.min(...vertices.map(point => point[0])) : 0;
                const maxX = vertices.length ? Math.max(...vertices.map(point => point[0])) : 1;
                const minY = vertices.length ? Math.min(...vertices.map(point => point[1])) : 0;
                const maxY = vertices.length ? Math.max(...vertices.map(point => point[1])) : 1;
                const scale = Math.min(168 / Math.max(maxX - minX, 0.01), 82 / Math.max(maxY - minY, 0.01));
                const panels = projected.map(polygon => {
                    const points = polygon.points.map(([x, y]) => `${(90 + (x - (minX + maxX) / 2) * scale).toFixed(2)},${(47 + (y - (minY + maxY) / 2) * scale).toFixed(2)}`).join(' ');
                    return `<polygon class="wing-thumbnail-panel" data-surface="${polygon.type}" points="${points}" />`;
                }).join('');
                return `<svg class="config-thumbnail-svg wing-thumbnail-svg" viewBox="0 0 180 94" role="img" aria-label="Aircraft lifting surfaces" focusable="false">
                    <path class="config-thumbnail-grid" d="M10 23H170 M10 47H170 M10 71H170" />
                    ${panels}
                </svg>`;
            },
            configItemHTML: (type, cfg) => {
                const filename = cfg.filename || (cfg.name ? `${cfg.name}.json` : 'unknown.json');
                const safeFilename = this._escapeHTML(filename);
                if (type === 'airfoil') {
                    const modified = this._escapeHTML(cfg.modified_at ? this._formatDate(cfg.modified_at) : '—');
                    const sizeKB = cfg.size ? `${(cfg.size / 1024).toFixed(1)} KB` : '—';
                    const airfoilMeta = [cfg.naca ? `NACA ${this._escapeHTML(cfg.naca)}` : null, cfg.chord != null ? `${this._escapeHTML(cfg.chord)} m chord` : null, cfg.points ? `${this._escapeHTML(cfg.points)} points` : null].filter(Boolean).join(' · ');
                    const desc = cfg.description ? `<div class="config-description">${this._escapeHTML(cfg.description)}</div>` : '';
                    const features = [];
                    if (cfg.has_airfoil) features.push('<span class="config-feature">Airfoil</span>');
                    if (cfg.has_coordinates) features.push('<span class="config-feature">Coordinates</span>');
                    const thumbnail = this.templates.airfoilThumbnailSVG(cfg);
                    return `
                        <div class="config-item card airfoil-card" data-filename="${safeFilename}">
                            <input class="config-selection-control" type="checkbox" aria-label="Select airfoil ${this._escapeHTML(filename.replace('.json',''))}">
                            <div class="card-left"><div class="card-thumb config-thumbnail">${thumbnail}</div></div>
                            <div class="card-body config-info">
                                <div class="config-header"><h4 class="config-name">${this._escapeHTML(filename.replace('.json',''))}</h4><div class="config-meta"><small>Modified: ${modified} • ${sizeKB}</small></div></div>
                                <div class="config-meta"><small>${airfoilMeta || 'Airfoil configuration'}</small></div>
                                <div class="config-features">${features.join(' ')}</div>
                                ${desc}
                            </div>
                                <div class="config-actions">
                                <button class="btn btn-sm btn-primary" title="Load this airfoil configuration" aria-label="Load airfoil configuration" data-tooltip="Load this airfoil configuration" data-action="load" data-filename="${safeFilename}" data-type="airfoil"><span class="material-symbols-outlined" aria-hidden="true" focusable="false">download</span></button>
                                <button class="btn btn-sm btn-danger" title="Delete this airfoil configuration" aria-label="Delete airfoil configuration" data-tooltip="Delete this airfoil configuration" data-action="delete" data-filename="${safeFilename}" data-type="airfoil"><span class="material-symbols-outlined" aria-hidden="true" focusable="false">delete</span></button>
                            </div>
                        </div>
                    `;
                }
                const modified = this._escapeHTML(cfg.modified_at ? this._formatDate(cfg.modified_at) : '—');
                const sizeKB = cfg.size ? (cfg.size / 1024).toFixed(1) + ' KB' : '—';
                const nameDisplay = this._escapeHTML(cfg.name ? cfg.name : filename.replace('.json',''));
                const descriptionDesc = cfg.description ? this._escapeHTML(cfg.description) : 'no description';
                const sectionCount = cfg.wing_sections_count || 0;
                const thumbnail = this.templates.wingThumbnailSVG(cfg);
                return `
                    <div class="config-item card modern-card result-card" data-filename="${safeFilename}">
                        <input class="config-selection-control" type="checkbox" aria-label="Select configuration ${nameDisplay}">
                        <div class="card-left">
                            <div class="card-thumb config-thumbnail">${thumbnail}</div>
                        </div>
                        <div class="card-body config-info">
                            <div class="config-row">
                                <div class="config-title">
                                    <h4 class="config-name">${this._escapeHTML(nameDisplay)}</h4>
                                    <div class="config-filename">${safeFilename}</div>
                                </div>
                                <div class="config-meta">
                                    <div class="meta-top"><small>Sections: ${sectionCount} • ${sizeKB}</small></div>
                                    <div class="meta-bottom"><small>Modified: ${modified}</small></div>
                                </div>
                            </div>
                            <div class="config-description">${descriptionDesc}</div>
                            <div class="config-features">${cfg.has_wing ? '<span class="config-feature">Wing</span>' : ''}${cfg.has_horizontal_stabilizer ? '<span class="config-feature">H-Stab</span>' : ''}${cfg.has_vertical_stabilizer ? '<span class="config-feature">V-Stab</span>' : ''}</div>
                        </div>
                        <div class="config-actions">
                            <button type="button" class="btn btn-sm btn-primary" title="Load this plane configuration" data-tooltip="Load this plane configuration" data-action="load" data-filename="${safeFilename}" data-type="plane" aria-label="Load plane configuration">
                                <span class="material-symbols-outlined" aria-hidden="true" focusable="false">download</span>
                            </button>
                            <button type="button" class="btn btn-sm btn-danger" title="Delete this plane configuration" data-tooltip="Delete this plane configuration" data-action="delete" data-filename="${safeFilename}" data-type="plane" aria-label="Delete plane configuration">
                                <span class="material-symbols-outlined" aria-hidden="true" focusable="false">delete</span>
                            </button>
                        </div>
                    </div>
                `;
            },
            resultItemHTML: (res) => {
                const safeFilename = this._escapeHTML(res.filename || 'unknown.json');
                const modified = this._escapeHTML(this._formatDate(res.modified_at));
                const rawCL = res.summary?.CL;
                const rawCD = res.summary?.CD;
                const cl = (rawCL === null || rawCL === undefined) ? '—' : Number(rawCL).toFixed(4);
                const cd = (rawCD === null || rawCD === undefined) ? '—' : Number(rawCD).toFixed(6);
                const displayName = this._escapeHTML(res.name ? String(res.name) : safeFilename.replace(/\.json$/i, ''));
                const showFilename = displayName !== safeFilename.replace(/\.json$/i, '');
                return `
                    <div class="config-item card modern-card result-card" data-filename="${safeFilename}">
                        <div class="card-left config-select"><input class="config-selection-control" type="checkbox" aria-label="Select results ${displayName}"></div>
                        <div class="card-body config-info">
                            <div class="config-row">
                                <div class="config-title">
                                    <h4 class="config-name">${displayName}</h4>
                                    ${showFilename ? `<div class="config-filename">${safeFilename}</div>` : ''}
                                </div>
                                <div class="config-meta">
                                    <div class="meta-top"><small>CL: ${cl} • CD: ${cd}</small></div>
                                    <div class="meta-bottom"><small>Modified: ${modified}</small></div>
                                </div>
                            </div>
                        </div>
                        <div class="config-actions">
                            <button type="button" class="btn btn-sm btn-primary" title="Load this results set" data-tooltip="Load this results set" data-action="load-results" data-filename="${safeFilename}" data-type="results" aria-label="Load results">
                                <span class="material-symbols-outlined" aria-hidden="true" focusable="false">download</span>
                            </button>
                            <button type="button" class="btn btn-sm btn-danger" title="Delete this results set" data-tooltip="Delete this results set" data-action="delete-results" data-filename="${safeFilename}" data-type="results" aria-label="Delete results">
                                <span class="material-symbols-outlined" aria-hidden="true" focusable="false">delete</span>
                            </button>
                        </div>
                    </div>`;
            },
            unifiedListHTML: (type, items) => {
                const body = items && items.length ? items.map(item => {
                    if (type === 'results') return this.templates.resultItemHTML(item);
                    return this.templates.configItemHTML(type === 'airfoil' ? 'airfoil' : 'plane', item);
                }).join('') : '<div class="empty-state">No items found.</div>';
                return `<div class="unified-list modern"><div class="config-list">${body}<div class="config-search-empty" hidden>No matching items.</div></div></div>`;
            },
            configListHTML: (type, configs) => {
                if (!configs || configs.length === 0) return '<div class="empty-state">No saved configurations found.</div>';
                return `<div class="config-list">${configs.map(cfg => this.templates.configItemHTML(type, cfg)).join('')}</div>`;
            }
        };
        // Event bindings
        this._handleGlobalClick = this._handleGlobalClick.bind(this);
        this._handleGlobalKey = this._handleGlobalKey.bind(this);
    }


    // -----------------------
    // Helpers / Utilidades
    // -----------------------
    _createContainer(id) {
        const c = document.createElement('div');
        c.id = id;
        document.body.appendChild(c);
        return c;
    }
    _formatDate(input) {
        try { return new Date(input).toLocaleString(); } catch { return input || ''; }
    }
    // Normaliza endpoint para asegurar prefijo slash
    _normalizeEndpoint(endpoint) {
        if (!endpoint) return '/';
        return endpoint.startsWith('/') ? endpoint : '/' + endpoint;
    }
    // API call unificada (usa cache y abort por endpoint)
    async apiCall(endpoint, opts = {}) {
        const ep = this._normalizeEndpoint(endpoint);
        const method = (opts.method || 'GET').toUpperCase();
        const bodyKey = opts.body ? `:${typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body)}` : '';
        const cacheKey = `${method}:${ep}${method === 'GET' ? bodyKey : ''}`;
        if (method === 'GET' && this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        try {
            const data = window.vlmpyCore
                ? await window.vlmpyCore.request(`${this.apiBase}${ep}`, {method, headers: opts.headers, body: opts.body})
                : await fetch(`${this.apiBase}${ep}`, {method, body: typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body)})
                    .then(async response => {
                        const value = await response.json();
                        if (!response.ok) throw new Error(value.message || `HTTP ${response.status}`);
                        return value;
                    });
            if (method === 'GET') this.cache.set(cacheKey, data);
            return data;
        } catch (err) {
            console.error('apiCall error', ep, err);
            throw err;
        }
    }

    // -----------------------
    // Messages and modals.
    // -----------------------
    showMessage(text, type = 'info', timeout = this.config.messageTimeout) {
        if (window.messageCenter) {
            // Map types if needed.
            const messageType = type.toLowerCase();
            if (typeof window.messageCenter[messageType] === 'function') {
                window.messageCenter[messageType](text, timeout > 0 ? timeout : undefined);
            } else {
                window.messageCenter.info(text, timeout > 0 ? timeout : undefined);
            }
        } else {
            // Fall back if MessageCenter is unavailable.
            console.warn('MessageCenter not available, message:', text);
            alert(text); // Basic fallback.
        }
    }
    showModal(html) {
        this.closeAllModals(true);
        document.body.insertAdjacentHTML('beforeend', html);
        // Delegate modal clicks for closing and internal actions.
        const modal = document.querySelector('[data-modal="true"]');
        if (modal) {
            modal.addEventListener('click', (e) => {
                const btn = e.target.closest('[data-action]');
                if (!btn) return;
                const action = btn.dataset.action;
                if (action === 'close') this.closeAllModals();
                // Save and other actions are handled by the global listener (see init).
            });
            // Close when clicking the modal overlay.
            modal.addEventListener('click', (e) => { if (e.target === modal) this.closeAllModals(); });
            // Double-click a card to load it.
            modal.addEventListener('dblclick', (e) => {
                try {
                    if (e.target.closest('[data-action]')) return; // ignore double-clicks on action buttons
                    const item = e.target.closest('.config-item');
                    if (!item) return;
                    const fname = item.dataset.filename;
                    // Use executeLoadResults when the item has a load-results button.
                    const hasLoadResults = !!item.querySelector('[data-action="load-results"]');
                    if (hasLoadResults) {
                        if (typeof this.executeLoadResults === 'function') this.executeLoadResults(fname);
                        return;
                    }
                    // Determine the item type.
                    const explicit = item.querySelector('[data-action][data-type]');
                    const type = explicit ? explicit.dataset.type : this.currentType;
                    if (fname) this.executeLoad(fname, type);
                } catch (err) { console.warn('dblclick load handler error', err); }
            });
        }
    }
    closeAllModals(immediate = false) {
        document.querySelectorAll('[data-modal]').forEach(m => {
            if (immediate) { m.remove(); return; }
            if (m.dataset.modal === 'closing') return;
            // Keep [data-modal="true"] selectors from treating a closing modal as active.
            m.dataset.modal = 'closing';
            m.setAttribute('aria-hidden', 'true');
            setTimeout(() => m.remove(), 180);
        });
    }

    // Snackbar / micro-feedback
    showSnackbar(message, timeout = 3000) {
        let el = document.getElementById('vlmpy-snackbar');
        if (!el) {
            el = document.createElement('div');
            el.id = 'vlmpy-snackbar';
            el.className = 'vlmpy-snackbar';
            document.body.appendChild(el);
        }
        el.textContent = message;
        el.classList.add('show');
        clearTimeout(el._hideTimeout);
        el._hideTimeout = setTimeout(() => {
            el.classList.remove('show');
        }, timeout);
    }

    _escapeHTML(value) {
        return String(value ?? '').replace(/[&<>"']/g, character => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[character]));
    }

    // -----------------------
    // Configuration management.
    // -----------------------
    // List configurations; the type determines the endpoint.
    async listConfigs(type) {
        try {
            if (type === 'airfoil') {
                const res = await this.apiCall('/airfoil-configs', { method: 'GET' });
                return res?.configs || [];
            } else {
                const res = await this.apiCall('/configs', { method: 'GET' });
                return res?.configs || [];
            }
        } catch (err) {
            this.showMessage(`Error listing configs: ${err.message}`, 'error');
            return null;
        }
    }

    // Open the save modal; type determines the endpoint and form.
    async saveConfigAdvanced(type) {
        if (type === 'plane' && this.elements.wingForm && !this.elements.wingForm.checkValidity()) {
            this.elements.wingForm.reportValidity();
            return;
        }
        this.currentType = type;
        const footer = `
            <button type="button" class="btn btn-secondary" data-action="close">Cancel</button>
            <button type="button" class="btn btn-primary" data-action="save-config">Save</button>
        `;
        const html = this.templates.modalHTML('Save Configuration', this.templates.saveForm(type), footer);
        this.showModal(html);
        // Focus the first input without blocking.
        requestAnimationFrame(() => {
            const inp = document.querySelector('#save-config-form input[type="text"]');
            if (inp) inp.focus();
        });
    }

    // Execute the save action (called by the delegated click handler).
    async executeSave(button) {
        const form = document.getElementById('save-config-form');
        if (!form) return this.showMessage('Save form not found', 'error');
        if (!form.checkValidity()) return form.reportValidity();
        const fd = new FormData(form);
        const payload = { name: fd.get('name'), filename: fd.get('filename'), description: fd.get('description') || '' };
        if (this.currentType === 'airfoil') {
            const additional = this._collectAirfoilFormData();
            payload.data = { ...(payload.data || {}), ...additional };
        }
        const btn = button || document.querySelector('[data-action="save-config"]');
        const prevText = btn ? btn.textContent : null;
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Saving...';
        }
        try {
            if (this.currentType === 'airfoil') {
                const result = await this.apiCall('/saveairfoilconfig', { method: 'POST', body: payload });
                this.showMessage(result?.replaced_existing ? 'Airfoil replaced; backup created.' : (result?.message || 'Airfoil saved'), 'success');
            } else {
                if (this.elements.wingForm) {
                    const wingData = new FormData(this.elements.wingForm);
                    if (window.vlmpyCore) {
                        await window.vlmpyCore.request('/wing', { method: 'POST', body: wingData });
                    } else {
                        const response = await fetch('/wing', { method: 'POST', body: wingData });
                        const result = await response.json();
                        if (!response.ok || result.status !== 'success') throw new Error(result.message || 'Wing computation failed');
                    }
                }
                const result = await this.apiCall('/save_plane_config', { method: 'POST', body: payload });
                this.showMessage(result?.replaced_existing ? 'Plane configuration replaced; backup created.' : (result?.message || 'Plane config saved'), 'success');
            }
            this.closeAllModals();
            this.cache.clear();
        } catch (err) {
            this.showMessage(`Save failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.disabled = false; if (prevText !== null) btn.textContent = prevText; }
        }
    }

    // Open the load modal and list configurations.
    async loadConfigAdvanced(type) {
        this.currentType = type;
        const configs = await this.listConfigs(type);
        if (!configs) return;
        if (type === 'plane' && !configs.some(c => c.wing_sections_count || c.has_wing || c.has_horizontal_stabilizer || c.has_vertical_stabilizer)) {
            const footer = '<button type="button" class="btn btn-secondary" data-action="close">Close</button>';
            this.showModal(this.templates.modalHTML('Load Configuration', '<div class="empty-state">No wings saved yet.</div>', footer));
            return;
        }
        if (!configs.length) { this.showMessage('No saved configurations found', 'warning'); return; }
        // For 'plane' view we must hide airfoil entries entirely: only show plane-like configs
        let useConfigs = configs;
        if (type === 'plane') {
            useConfigs = configs.filter(c => c.wing_sections_count || c.has_wing || c.has_horizontal_stabilizer || c.has_vertical_stabilizer);
        }
        const listHTML = this.templates.unifiedListHTML(type, useConfigs);
        const headerHTML = `
            <div class="config-list-header compact">
                <div class="header-left">
                    <input id="config-search-input" class="input input-sm" placeholder="Search..." aria-label="Search saved items" />
                </div>
                <div class="header-center">
                    <button id="select-all-button" class="btn btn-sm btn-secondary">Select all</button>
                </div>
                <div class="header-right">
                    <button class="btn btn-sm btn-danger" id="bulk-delete-button" title="Delete selected items" data-tooltip="Delete selected items" data-action="bulk-delete" data-type="${type}">
                        Delete selected
                    </button>
                </div>
            </div>`;
        const footer = `<button type="button" class="btn btn-secondary" data-action="close">Cancel</button>`;
        this.showModal(this.templates.modalHTML('Load Configuration', listHTML, footer, headerHTML));
            requestAnimationFrame(() => {
                const modal = document.querySelector('[data-modal="true"]');
                const selectAllBtn = modal && modal.querySelector('#select-all-button');
                const setSelected = (item, selected) => {
                    item.classList.toggle('selected', selected);
                    const checkbox = item.querySelector('.config-selection-control');
                    if (checkbox) checkbox.checked = selected;
                };
                if (selectAllBtn) selectAllBtn.addEventListener('click', (e) => {
                    const visible = Array.from(modal.querySelectorAll('.config-item')).filter(i => i.style.display !== 'none');
                    const anyUnselected = visible.some(c => !c.classList.contains('selected'));
                    visible.forEach(card => setSelected(card, anyUnselected));
                    selectAllBtn.textContent = anyUnselected ? 'Unselect all' : 'Select all';
                });
                // Search/filter
                const search = modal && modal.querySelector('#config-search-input');
                if (search) {
                    search.addEventListener('input', (ev) => {
                        const term = String(ev.target.value || '').toLowerCase();
                        const items = Array.from(modal.querySelectorAll('.config-item'));
                        let visibleCount = 0;
                        items.forEach(item => {
                            const visible = item.textContent.toLowerCase().includes(term);
                            item.style.display = visible ? '' : 'none';
                            if (visible) visibleCount += 1;
                        });
                        const emptyState = modal.querySelector('.config-search-empty');
                        if (emptyState) emptyState.hidden = !term || visibleCount > 0;
                    });
                }
                // Clicking an item toggles selected visual state (but not a preview)
                modal && modal.querySelectorAll('.config-item').forEach(item => {
                    item.tabIndex = 0;
                    const checkbox = item.querySelector('.config-selection-control');
                    checkbox?.addEventListener('change', () => setSelected(item, checkbox.checked));
                    item.addEventListener('keydown', event => {
                        if (event.target !== item || ![' ', 'Enter'].includes(event.key)) return;
                        event.preventDefault();
                        setSelected(item, !item.classList.contains('selected'));
                    });
                    item.addEventListener('click', (ev) => {
                        if (ev.target.closest('[data-action], .config-selection-control')) return;
                        setSelected(item, !item.classList.contains('selected'));
                    });
                });
            });
    }

    // Execute the load action (called by the delegated click handler).
    async executeLoad(filename, type) {
        if (!filename) return this.showMessage('Filename missing', 'error');
        try {
            if (type === 'airfoil') {
                const result = await this.apiCall('/loadairfoilconfig', { method: 'POST', body: { filename } });
                if (result?.status === 'success' && result.airfoil) {
                    const populated = this._populateAirfoilForm(result.airfoil);
                    if (!populated) throw new Error('Airfoil form is not available');
                    this.clearPlotArea('airfoil');
                    if (typeof clearPlots === 'function') try { clearPlots(); } catch (e) { console.warn('clearPlots threw', e); }
                    window.dispatchEvent(new CustomEvent('vlmpy:airfoil-loaded', {detail: result.airfoil}));
                    this.showMessage(result?.message || 'Airfoil config loaded', 'success');
                } else {
                    this.showMessage(result?.message || 'Failed to load airfoil config', 'error');
                    throw new Error(result?.message || 'Failed to load airfoil config');
                }
            } else {
                const result = await this.apiCall('/load_plane_config', { method: 'POST', body: { filename, loadflightparams: true, verifychecksum: true } });
                this.showMessage(result?.message || 'Loaded', result?.status === 'success' ? 'success' : 'error');
                if (result && result.checksum_valid === false) {
                    this.showMessage('Warning: Invalid checksum', 'warning');
                }
                if (result?.status === 'success' && result.config) {
                    if (typeof window.populateFormWithConfig === 'function') {
                        const ok = await window.populateFormWithConfig(result.config);
                        if (!ok) this.showMessage('Error updating form', 'error');
                        if (ok) {
                            this.clearPlotArea('plane');
                            if (typeof clearPlots === 'function') try { clearPlots(); } catch (e) { console.warn('clearPlots threw', e); }
                        }
                    } else {
                        this._populateWingForm(result.config);
                        this.clearPlotArea('plane');
                        if (typeof clearPlots === 'function') try { clearPlots(); } catch (e) { console.warn('clearPlots threw', e); }
                    }
                }
            }
            this.closeAllModals();
        } catch (err) {
            console.error('Error:', err);
            this.showMessage(`Error loading plane configuration: ${err.message}`, 'error');
        }
    }

    async saveResultsAdvanced() {
        const body = `
            <form id="save-results-form" class="form">
                <div class="form-group">
                    <label for="results-config-filename">Filename:</label>
                    <input type="text" id="results-config-filename" name="filename"
                        placeholder="results_case" pattern="^[a-zA-Z0-9_-]+$" required>
                </div>
            </form>`;
        const footer = `
            <button type="button" class="btn btn-secondary" data-action="close">Cancel</button>
            <button type="button" class="btn btn-primary" data-action="save-results">Save</button>`;
        this.showModal(this.templates.modalHTML('Save Results', body, footer));
    }

    async loadResultsAdvanced() {
        try {
            const response = await this.apiCall('/results', {method: 'GET'});
            const results = response?.results || [];
            if (!results.length) return this.showMessage('No saved results found', 'warning');
            const listHTML = this.templates.unifiedListHTML('results', results);
            const headerHTML = `
                <div class="config-list-header compact">
                    <div class="header-left">
                        <input id="config-search-input" class="input input-sm" placeholder="Search..." aria-label="Search saved items" />
                    </div>
                    <div class="header-center">
                        <button id="select-all-button" class="btn btn-sm btn-secondary">Select all</button>
                    </div>
                    <div class="header-right">
                        <button class="btn btn-sm btn-danger" id="bulk-delete-button" title="Delete selected items" data-tooltip="Delete selected items" data-action="bulk-delete" data-type="results">
                            Delete selected
                        </button>
                    </div>
                </div>`;
            this.showModal(this.templates.modalHTML('Load Results', listHTML,
                '<button type="button" class="btn btn-secondary" data-action="close">Cancel</button>', headerHTML));
            requestAnimationFrame(() => {
                const modal = document.querySelector('[data-modal="true"]');
                const selectAllBtn = modal && modal.querySelector('#select-all-button');
                const setSelected = (item, selected) => {
                    item.classList.toggle('selected', selected);
                    const checkbox = item.querySelector('.config-selection-control');
                    if (checkbox) checkbox.checked = selected;
                };
                if (selectAllBtn) selectAllBtn.addEventListener('click', (e) => {
                    const visible = Array.from(modal.querySelectorAll('.config-item')).filter(i => i.style.display !== 'none');
                    const anyUnselected = visible.some(c => !c.classList.contains('selected'));
                    visible.forEach(card => setSelected(card, anyUnselected));
                    selectAllBtn.textContent = anyUnselected ? 'Unselect all' : 'Select all';
                });
                const search = modal && modal.querySelector('#config-search-input');
                if (search) {
                    search.addEventListener('input', (ev) => {
                        const term = String(ev.target.value || '').toLowerCase();
                        const items = Array.from(modal.querySelectorAll('.config-item'));
                        let visibleCount = 0;
                        items.forEach(item => {
                            const visible = item.textContent.toLowerCase().includes(term);
                            item.style.display = visible ? '' : 'none';
                            if (visible) visibleCount += 1;
                        });
                        const emptyState = modal.querySelector('.config-search-empty');
                        if (emptyState) emptyState.hidden = !term || visibleCount > 0;
                    });
                }
                modal && modal.querySelectorAll('.config-item').forEach(item => {
                    item.tabIndex = 0;
                    const checkbox = item.querySelector('.config-selection-control');
                    checkbox?.addEventListener('change', () => setSelected(item, checkbox.checked));
                    item.addEventListener('keydown', event => {
                        if (event.target !== item || ![' ', 'Enter'].includes(event.key)) return;
                        event.preventDefault();
                        setSelected(item, !item.classList.contains('selected'));
                    });
                    item.addEventListener('click', (ev) => {
                        if (ev.target.closest('[data-action], .config-selection-control')) return;
                        setSelected(item, !item.classList.contains('selected'));
                    });
                });
            });
        } catch (error) {
            this.showMessage(`Error listing results: ${error.message}`, 'error');
        }
    }

    async executeSaveResults() {
        const input = document.getElementById('results-config-filename');
        if (!input?.checkValidity()) return input?.reportValidity();
        try {
            const result = await this.apiCall('/results/save', {
                method: 'POST', body: {filename: input.value}
            });
            this.showMessage(result?.message || `Results saved as ${result.filename}`, 'success');
            this.closeAllModals();
            this.cache.clear();
        } catch (error) {
            this.showMessage(`Save results failed: ${error.message}`, 'error');
        }
    }

    async executeLoadResults(filename) {
        try {
            const result = await this.apiCall('/results/load', {
                method: 'POST', body: {filename}
            });
            if (result?.status !== 'success') throw new Error(result?.message || 'Load failed');
            const summary = result.summary || {};
            const values = {
                'result-cl': summary.CL,
                'result-cd': summary.CD,
                'result-lift': summary.lift,
                'result-drag': summary.drag
            };
            Object.entries(values).forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element && value != null) {
                    element.textContent = ['result-lift', 'result-drag'].includes(id)
                        ? Number(value).toFixed(2)
                        : Number(value).toFixed(4);
                }
            });
            this.clearPlotArea('plane', false);
            this.closeAllModals();
            this.showMessage(`Results loaded from ${filename}`, 'success');
        } catch (error) {
            this.showMessage(`Load results failed: ${error.message}`, 'error');
        }
    }

    async executeBulkDelete(type) {
        // collect selected filenames
        const cards = Array.from(document.querySelectorAll('.config-item.selected'));
        if (!cards.length) return this.showMessage('No items selected', 'warning');
        const filenames = cards.map(c => c.dataset.filename).filter(Boolean);
        if (!filenames.length) return this.showMessage('No valid filenames selected', 'error');
        if (!confirm(`Delete ${filenames.length} item(s)? This action cannot be undone.`)) return;
        // perform deletes grouped by type
        const results = [];
        for (const fname of filenames) {
            try {
                if (type === 'airfoil') {
                    const res = await this.apiCall('/deleteairfoilconfig', { method: 'POST', body: { filename: fname } });
                    results.push(res);
                } else if (type === 'plane') {
                    const res = await this.apiCall('/delete_config', { method: 'DELETE', body: { filename: fname } });
                    results.push(res);
                } else if (type === 'results') {
                    const res = await this.apiCall('/results', { method: 'DELETE', body: { filename: fname } });
                    results.push(res);
                } else {
                    results.push({ status: 'error', message: 'Unknown type' });
                }
            } catch (err) {
                results.push({ status: 'error', message: err.message });
            }
        }
        // report outcome and refresh modal
        const failures = results.filter(r => r?.status !== 'success');
        if (failures.length === 0) {
            this.showMessage('Selected items deleted', 'success');
            this.showSnackbar(`Deleted ${filenames.length} item(s)`);
        } else {
            this.showMessage(`${failures.length} failures during delete`, 'warning');
            this.showSnackbar(`${failures.length} failures during delete`, 4000);
        }
        // refresh current modal
        if (type === 'results') await this.loadResultsAdvanced(); else await this.loadConfigAdvanced(type);
    }

    async executeDeleteResults(filename) {
        if (!filename) return this.showMessage('Filename missing', 'error');
        if (!confirm(`Delete results "${filename}"?`)) return;
        try {
            const result = await this.apiCall('/results', { method: 'DELETE', body: { filename } });
            if (result?.status === 'success') {
                this.showMessage(result.message || `Deleted ${filename}`, 'success');
                this.showSnackbar(`Deleted ${filename}`);
                // Refresh modal list if open
                if (document.querySelector('[data-modal="true"]')) await this.loadResultsAdvanced();
            } else {
                this.showMessage(result?.message || 'Delete failed', 'error');
            }
        } catch (err) {
            this.showMessage(`Delete results failed: ${err.message}`, 'error');
        }
    }

    // Delete a configuration.
    async deleteConfig(filename, type) {
        if (!filename) return;
        if (!confirm(`Delete "${filename}"?`)) return;
        try {
            let data;
            if (type === 'airfoil') {
                data = await this.apiCall('/deleteairfoilconfig', { method: 'POST', body: { filename } });
            } else {
                data = await this.apiCall('/delete_config', { method: 'DELETE', body: { filename } });
            }
            this.showMessage(data.message, data.status === 'success' ? 'success' : 'error');
            if (data?.status === 'success') this.showSnackbar(`Deleted ${filename}`);
            this.cache.clear();
            if (document.querySelector('[data-modal="true"]')) await this.loadConfigAdvanced(type);
        } catch (err) {
            this.showMessage(`Delete error: ${err.message}`, 'error');
        }
    }

    // Validate the current configuration based on its type.
    async validateCurrentConfig(type) {
        if (type === 'airfoil') {
            const data = this._collectAirfoilFormData();
            try {
                const result = await this.apiCall('/validateairfoilconfig', { method: 'POST', body: { data } });
                if (result?.valid) this.showMessage('Configuration is valid!', 'success');
                else this.showMessage(`Validation failed: ${result.errors?.join(', ') || result.message}`, 'error');
            } catch (err) { this.showMessage(err.message, 'error'); }
        } else {
            if (!this.elements.wingForm) return this.showMessage('Form not found', 'error');
            try {
                const payload = Object.fromEntries(new FormData(this.elements.wingForm).entries());
                const result = await this.apiCall('/validate_config', { method: 'POST', body: payload });
                if (result?.is_valid) this.showMessage('Configuration is valid', 'success');
                else this.showMessage(result?.message || 'Invalid configuration', 'warning');
            } catch (err) { this.showMessage(`Validation error: ${err.message}`, 'error'); }
        }
    }

    // -----------------------
    // Form helpers (airfoil + wing)
    // -----------------------
    _collectAirfoilFormData() {
        if (!this.elements.airfoilForm) return {};
        const formData = new FormData(this.elements.airfoilForm);
        const data = {};
        for (const [key, value] of formData.entries()) {
            if (['u', 'alpha', 'chord', 'n'].includes(key)) { data[key] = parseFloat(value) || 0; }
            else { data[key] = value; }
        }
        return data;
    }
    _populateAirfoilForm(airfoilData) {
        if (!this.elements.airfoilForm) return false;
        const fieldMap = { 'naca': 'naca', 'u': 'u', 'alpha': 'alpha', 'chord': 'chord', 'n': 'n' };
        Object.entries(fieldMap).forEach(([formField, dataField]) => {
            const el = document.getElementById(formField);
            if (el && airfoilData[dataField] !== undefined) el.value = airfoilData[dataField];
        });
        const naca = String(airfoilData.naca || '').padStart(4, '0').slice(0, 4);
        ['naca-camber', 'naca-position', 'naca-thickness', 'naca-thickness-last'].forEach((id, index) => {
            const slider = document.getElementById(id);
            const output = document.getElementById(`${id}-value`);
            if (slider) slider.value = naca[index] || '0';
            if (output) output.textContent = naca[index] || '0';
        });
        window.requestAnimationFrame(() => window.renderAirfoilEditor?.());
        return true;
    }
    _populateWingForm(cfg) {
        if (!this.elements.wingForm) return false;
        try {
            Object.entries(cfg).forEach(([k, v]) => {
                const el = this.elements.wingForm.querySelector(`[name="${k}"]`) || document.getElementById(k);
                if (el) {
                    if (el.type === 'checkbox') el.checked = Boolean(v);
                    else el.value = (v === null || v === undefined) ? '' : String(v);
                }
            });
            return true;
        } catch (err) { console.warn('populateWingForm failed', err); return false; }
    }
    clearPlotArea(type, showMessage = true) {
        if (!this.elements.plotArea) return;
        if (window.Plotly && this.elements.plotArea.classList.contains('js-plotly-plot')) {
            Plotly.purge(this.elements.plotArea);
        }
        if (!showMessage) {
            this.elements.plotArea.replaceChildren();
            return;
        }
        const message = document.createElement('p');
        message.className = 'config-loaded-message';
        message.textContent = type === 'airfoil'
            ? 'New airfoil configuration loaded. Click Calculate to see updated results.'
            : 'New plane configuration loaded. Click Calculate to see updated results.';
        this.elements.plotArea.replaceChildren(message);
    }

    // -----------------------
    // Eventos globales (delegados)
    // -----------------------
    _handleGlobalClick(e) {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;
        e.preventDefault();
        const action = btn.dataset.action;
        const filename = btn.dataset.filename;
        const type = btn.dataset.type || this.currentType;
        switch (action) {
            case 'close': this.closeAllModals(); break;
            case 'save-config': this.executeSave(btn); break;
            case 'save-results': this.executeSaveResults(); break;
            case 'open-save': this.saveConfigAdvanced(btn.dataset.type || 'plane'); break;
            case 'load': this.executeLoad(filename, type); break;
            case 'load-results': this.executeLoadResults(filename); break;
            case 'delete-results': this.executeDeleteResults(filename); break;
            case 'bulk-delete': this.executeBulkDelete(btn.dataset.type || this.currentType); break;
            case 'delete': this.deleteConfig(filename, type); break;
            case 'saveAirfoilAdvanced': case 'savePlaneAdvanced':
                if (action === 'saveAirfoilAdvanced') this.saveConfigAdvanced('airfoil'); else this.saveConfigAdvanced('plane');
                break;
            case 'validateConfig': case 'validateAirfoilConfig':
                if (action === 'validateAirfoilConfig') this.validateCurrentConfig('airfoil'); else this.validateCurrentConfig('plane');
                break;
            default: break;
        }
    }
    _handleGlobalKey(e) { if (e.key === 'Escape') this.closeAllModals(); }

    // -----------------------
    // Backwards compatibility: exponer funciones globales
    // -----------------------
    _exposeGlobals() {
        const map = {
            saveAirfoilAdvanced: () => this.saveConfigAdvanced('airfoil'),
            loadAirfoilAdvanced: () => this.loadConfigAdvanced('airfoil'),
            validateAirfoilConfig: () => this.validateCurrentConfig('airfoil'),
            savePlaneAdvanced: () => this.saveConfigAdvanced('plane'),
            loadPlaneAdvanced: () => this.loadConfigAdvanced('plane'),
            validateConfig: () => this.validateCurrentConfig('plane')
            ,saveResultsAdvanced: () => this.saveResultsAdvanced()
            ,loadResultsAdvanced: () => this.loadResultsAdvanced()
        };
        Object.entries(map).forEach(([k, v]) => { window[k] = v; });
    }

    // -----------------------
    // Introspection and state.
    // -----------------------
    getStatus() {
        return { initialized: true, cacheSize: this.cache.size, domFound: { messageContainer: !!this.elements.messageContainer, plotArea: !!this.elements.plotArea, airfoilForm: !!this.elements.airfoilForm, wingForm: !!this.elements.wingForm }, debug: this.config.debug };
    }

    // -----------------------
    // Initialization.
    // -----------------------
    async init() {
        document.addEventListener('click', this._handleGlobalClick);
        document.addEventListener('keydown', this._handleGlobalKey);
        this._exposeGlobals();
        this.log('UnifiedConfigManager initialized');
        try {
            await Promise.allSettled([
                this.apiCall('/airfoil-configs', { method: 'GET' }).catch(() => null),
                this.apiCall('/configs', { method: 'GET' }).catch(() => null)
            ]);
        } catch {}
        return true;
    }
    log(...args) { if (this.config.debug) console.log('[UnifiedConfigManager]', ...args); }
}

// Automatic initialization with a fallback.
let unifiedConfigManager = null;
document.addEventListener('DOMContentLoaded', async () => {
    try {
        unifiedConfigManager = new UnifiedConfigManager();
        await unifiedConfigManager.init();
        window.unifiedConfigManager = unifiedConfigManager;
        // Also ensure common global helpers point to the instance methods
        window.saveResultsAdvanced = () => unifiedConfigManager.saveResultsAdvanced();
        window.loadResultsAdvanced = () => unifiedConfigManager.loadResultsAdvanced();
        window.saveAirfoilAdvanced = () => unifiedConfigManager.saveConfigAdvanced('airfoil');
        window.loadAirfoilAdvanced = () => unifiedConfigManager.loadConfigAdvanced('airfoil');
        window.savePlaneAdvanced = () => unifiedConfigManager.saveConfigAdvanced('plane');
        window.loadPlaneAdvanced = () => unifiedConfigManager.loadConfigAdvanced('plane');
        console.log('UnifiedConfigManager initialized', unifiedConfigManager.getStatus());
    } catch (err) {
        console.error('UnifiedConfigManager init error', err);
        const showError = (msg) => {
            if (window.messageCenter && typeof window.messageCenter.error === 'function') {
                window.messageCenter.error(msg, 15000);
            } else if (unifiedConfigManager && typeof unifiedConfigManager.showMessage === 'function') {
                unifiedConfigManager.showMessage(msg, 'error', 15000);
            } else {
                const msgContainer = document.getElementById('message-container');
                if (msgContainer) {
                    const message = document.createElement('div');
                    message.className = 'message error';
                    message.textContent = msg;
                    msgContainer.replaceChildren(message);
                } else {
                    // fallback
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'message error';
                    errorDiv.textContent = msg;
                    document.body.appendChild(errorDiv);
                    setTimeout(() => errorDiv.remove(), 15000);
                }
            }
        };
        const errorFn = () => showError('UnifiedConfigManager failed to initialize properly');
        ['saveAirfoilAdvanced', 'loadAirfoilAdvanced', 'validateAirfoilConfig', 'savePlaneAdvanced', 'loadPlaneAdvanced', 'validateConfig', 'saveResultsAdvanced', 'loadResultsAdvanced'].forEach(name => { window[name] = errorFn; });
    }
});
