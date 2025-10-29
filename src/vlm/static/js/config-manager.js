/**
 * UnifiedConfigManager - Gestor unificado para Airfoil y Plane (Wing)
 * Combina funcionalidades de airfoil-config-manager.js y plane-config-manager.js
 *
 * Autor: Javier L. (unificado, actualizado)
 */
class UnifiedConfigManager {
    constructor() {
        this.apiBase = '/api';
        this.cache = new Map();
        this.controllers = new Map(); // AbortController por endpoint
        this.currentType = null; // 'airfoil' | 'plane' (se establece al abrir modales)
        // Elementos DOM (busca airfoil-form o wing-form)
        this.elements = {
            messageContainer: document.getElementById('message-container'),
            plotArea: document.getElementById('plot-area'),
            airfoilForm: document.getElementById('airfoil-form') || null,
            wingForm: document.getElementById('wing-form') || null
        };
        this.config = { messageTimeout: 5000, debug: false };
        // Plantillas (precompiladas -> retornan nodos o strings según proceda)
        this.templates = {
            // Modal base (string)
            modalHTML: (title, body, footer = '') => `
                <div class="modal-overlay" data-modal="true">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3 class="modal-title">${title}</h3>
                            <button class="modal-close" data-action="close" aria-label="Close">×</button>
                        </div>
                        <div class="modal-body">${body}</div>
                        ${footer ? `<div class="modal-footer">${footer}</div>` : ''}
                    </div>
                </div>
            `,
            // Formulario de guardado: genera distinto HTML según tipo
            saveForm: (type) => {
            const pattern = "^[a-zA-Z0-9_\\-]+$"; // Escapado doble
            if (type === 'airfoil') {
                return `
                    <form id="save-config-form" data-form="save-config" class="form">
                        <div class="form-group">
                            <label for="config-name">Configuration Name:</label>
                            <input type="text" id="config-name" name="name" class="input"
                                placeholder="e.g., My airfoil config" maxlength="80" required>
                        </div>
                        <div class="form-group">
                            <label for="config-filename">Filename:</label>
                            <input type="text" id="config-filename" name="filename" class="input"
                                placeholder="e.g., my_airfoil_config" pattern="${pattern}"
                                maxlength="50" required title="Only letters, numbers, - and _ allowed">
                        </div>
                        <div class="form-group">
                            <label for="config-description">Description (optional):</label>
                            <textarea id="config-description" name="description" rows="3"
                                    maxlength="200" class="textarea" placeholder="Brief description..."></textarea>
                        </div>
                    </form>
                `;
            } else {
                return `
                    <form id="save-config-form" data-form="save-config" class="form">
                        <div class="form-group">
                            <label for="config-name">Configuration Name:</label>
                            <input type="text" id="config-name" name="name" required placeholder="e.g., My wing configuration">
                        </div>
                        <div class="form-group">
                            <label for="config-filename">Filename:</label>
                            <input type="text" id="config-filename" name="filename" required
                                placeholder="e.g., my_configuration" pattern="${pattern}"
                                title="Only letters, numbers, - and _ allowed">
                        </div>
                        <div class="form-group">
                            <label for="config-description">Description (optional):</label>
                            <textarea id="config-description" name="description" rows="3"
                                    placeholder="Describe main characteristics..."></textarea>
                        </div>
                    </form>
                `;
                }
            },
            // Item de lista (se adapta según tipo)
            configItemHTML: (type, cfg) => {
                const filename = cfg.filename || (cfg.name ? `${cfg.name}.json` : 'unknown.json');
                if (type === 'airfoil') {
                    const modified = cfg.modified_at ? this._formatDate(cfg.modified_at) : '—';
                    const sizeKB = cfg.size ? (cfg.size / 1024).toFixed(1) + ' KB' : '—';
                    const desc = cfg.description ? `<div class="config-description">${cfg.description}</div>` : '';
                    const features = [];
                    if (cfg.has_airfoil) features.push('<span class="config-feature">Airfoil</span>');
                    if (cfg.has_coordinates) features.push('<span class="config-feature">Coordinates</span>');
                    return `
                        <div class="config-item card" data-filename="${filename}">
                            <div class="card-body config-info">
                                <div class="config-header">
                                    <h4 class="config-name">${filename.replace('.json','')}</h4>
                                    <div class="config-meta"><small>Modified: ${modified} • ${sizeKB}</small></div>
                                </div>
                                <div class="config-features">${features.join(' ')}</div>
                                ${desc}
                                <div class="config-actions">
                                    <button class="btn btn-sm btn-primary" data-action="load" data-filename="${filename}" data-type="airfoil">Load</button>
                                    <button class="btn btn-sm btn-danger" data-action="delete" data-filename="${filename}" data-type="airfoil">Delete</button>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    const modified = cfg.modified_at ? this._formatDate(cfg.modified_at) : '—';
                    const sizeKB = cfg.size ? (cfg.size / 1024).toFixed(1) + ' KB' : '—';
                    const nameDisplay = cfg.name ? cfg.name : filename.replace('.json','');
                    const descriptionDesc = cfg.description ? `<div class="config-description">${cfg.description}</div>` : 'no description';
                    const sectionCount = cfg.wing_sections_count || 0;
                    // <small>Modified: ${modified} | Version: ${cfg.version || '—'} | Sections: ${cfg.wing_sections_count || 0} | Size: ${sizeKB}</small>
                    return `
                        <div class="config-item" data-filename="${filename}">
                            <div class="config-info">
                                <h4>${nameDisplay} <small>(${filename})</small></h4>
                                <div class="config-details">
                                    <small>Modified: ${modified} | Sections: ${sectionCount} | Size: ${sizeKB}</small>
                                    <small>Description: ${descriptionDesc}</small>
                                </div>
                                <div class="config-features">
                                    ${cfg.has_wing ? '<span class="config-feature">Wing</span>' : ''}
                                    ${cfg.has_horizontal_stabilizer ? '<span class="config-feature">H-Stab</span>' : ''}
                                    ${cfg.has_vertical_stabilizer ? '<span class="config-feature">V-Stab</span>' : ''}
                                </div>
                            </div>
                            <div class="config-actions">
                                <button class="btn btn-sm btn-primary" data-action="load" data-filename="${filename}" data-type="plane">Load</button>
                                <button class="btn btn-sm btn-danger" data-action="delete" data-filename="${filename}" data-type="plane">Delete</button>
                            </div>
                        </div>
                    `;
                }
            },
            configListHTML: (type, configs) => {
                if (!configs || configs.length === 0) {
                    return '<div class="empty-state">No saved configurations found.</div>';
                }
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
        // Abort previous controller para este endpoint
        const prev = this.controllers.get(ep);
        if (prev) prev.abort();
        const controller = new AbortController();
        this.controllers.set(ep, controller);
        const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
        const fetchOpts = { method, headers, signal: controller.signal };
        if (opts.body) {
            // body puede venir como objeto o string; aseguramos string JSON
            fetchOpts.body = typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body);
        }
        try {
            const res = await fetch(`${this.apiBase}${ep}`, fetchOpts);
            const text = await res.text();
            const data = text ? JSON.parse(text) : {};
            if (!res.ok) {
                const message = data?.message || `HTTP ${res.status}`;
                throw new Error(message);
            }
            if (method === 'GET') this.cache.set(cacheKey, data);
            return data;
        } catch (err) {
            if (err.name === 'AbortError') return null;
            console.error('apiCall error', ep, err);
            throw err;
        } finally {
            if (this.controllers.get(ep) === controller) this.controllers.delete(ep);
        }
    }

    // -----------------------
    // Mensajes y modales
    // -----------------------
    showMessage(text, type = 'info', timeout = this.config.messageTimeout) {
        if (window.messageCenter) {
            // Mapear tipos si es necesario
            const messageType = type.toLowerCase();
            if (typeof window.messageCenter[messageType] === 'function') {
                window.messageCenter[messageType](text, timeout > 0 ? timeout : undefined);
            } else {
                window.messageCenter.info(text, timeout > 0 ? timeout : undefined);
            }
        } else {
            // Fallback si MessageCenter no está disponible
            console.warn('MessageCenter not available, message:', text);
            alert(text); // Fallback básico
        }
    }
    showModal(html) {
        this.closeAllModals();
        document.body.insertAdjacentHTML('beforeend', html);
        // delegado: clicks en modal para cerrar o acciones internas
        const modal = document.querySelector('[data-modal="true"]');
        if (modal) {
            modal.addEventListener('click', (e) => {
                const btn = e.target.closest('[data-action]');
                if (!btn) return;
                const action = btn.dataset.action;
                if (action === 'close') this.closeAllModals();
                // acciones de save y demás son manejadas por escucha global (ver init)
            });
            // cerrar al clicar overlay modal
            modal.addEventListener('click', (e) => { if (e.target === modal) this.closeAllModals(); });
        }
    }
    closeAllModals() { document.querySelectorAll('[data-modal="true"]').forEach(m => m.remove()); }

    // -----------------------
    // Gestión de configuraciones
    // -----------------------
    // Lista configs (tipo determina endpoint)
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
            return [];
        }
    }

    // Abre modal guardar (type define endpoint y formulario)
    async saveConfigAdvanced(type) {
        this.currentType = type;
        const footer = `
            <button type="button" class="btn btn-secondary" data-action="close">Cancel</button>
            <button type="button" class="btn btn-primary" data-action="save-config">Save</button>
        `;
        const html = this.templates.modalHTML('Save Configuration', this.templates.saveForm(type), footer);
        this.showModal(html);
        // focus input inicial sin bloquear
        requestAnimationFrame(() => {
            const inp = document.querySelector('#save-config-form input[type="text"]');
            if (inp) inp.focus();
        });
    }

    // Ejecutar guardado (invocado por click delegado)
    async executeSave(button) {
        const form = document.getElementById('save-config-form');
        if (!form) return this.showMessage('Save form not found', 'error');
        if (!form.checkValidity()) return form.reportValidity();
        const fd = new FormData(form);
        const payload = { name: fd.get('name'), filename: fd.get('filename'), description: fd.get('description') || '' };
        if (this.currentType === 'airfoil') {
            const additional = this._collectAirfoilFormData();
            payload.data = { ...(payload.data || {}), ...additional };
        } else if (this.currentType === 'plane') {
            if (this.elements.wingForm) {
                const full = Object.fromEntries(new FormData(this.elements.wingForm).entries());
                payload.data = { ...(payload.data || {}), ...full };
            }
        }
        const btn = button || document.querySelector('[data-action="save-config"]');
        const prevText = btn ? btn.textContent : null;
        if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Saving...'; }
        try {
            if (this.currentType === 'airfoil') {
                const result = await this.apiCall('/saveairfoilconfig', { method: 'POST', body: payload });
                this.showMessage(result?.message || 'Airfoil saved', 'success');
            } else {
                const result = await this.apiCall('/save_plane_config', { method: 'POST', body: payload });
                this.showMessage(result?.message || 'Plane config saved', 'success');
            }
            this.closeAllModals();
            this.cache.clear();
        } catch (err) {
            this.showMessage(`Save failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.disabled = false; if (prevText !== null) btn.textContent = prevText; }
        }
    }

    // Abre modal cargar (lista configs)
    async loadConfigAdvanced(type) {
        this.currentType = type;
        const configs = await this.listConfigs(type);
        if (!configs.length) { this.showMessage('No saved configurations found', 'warning'); return; }
        const listHTML = this.templates.configListHTML(type, configs);
        const footer = `<button type="button" class="btn btn-secondary" data-action="close">Cancel</button>`;
        this.showModal(this.templates.modalHTML('Load Configuration', listHTML, footer));
    }

    // Ejecutar carga (invocado por click delegado)
    async executeLoad(filename, type) {
        if (!filename) return this.showMessage('Filename missing', 'error');
        try {
            if (type === 'airfoil') {
                const result = await this.apiCall('/loadairfoilconfig', { method: 'POST', body: { filename } });
                if (result?.status === 'success' && result.airfoil) {
                    this._populateAirfoilForm(result.airfoil);
                    this.clearPlotArea('airfoil');
                    if (typeof clearPlots === 'function') try { clearPlots(); } catch (e) { console.warn('clearPlots threw', e); }
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
                    if (typeof populateFormWithConfig === 'function') {
                        const ok = populateFormWithConfig(result.config);
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
            this.showMessage('An error occurred while loading the VLM state.', 'error');
        }
    }

    // Borrar config
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
            this.cache.clear();
            if (document.querySelector('[data-modal="true"]')) await this.loadConfigAdvanced(type);
        } catch (err) {
            this.showMessage(`Delete error: ${err.message}`, 'error');
        }
    }

    // Validar configuración actual (según tipo)
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
    clearPlotArea(type) {
        if (!this.elements.plotArea) return;
        if (type === 'airfoil') {
            this.elements.plotArea.innerHTML = '<p style="color:#666;text-align:center;padding:20px;">New airfoil configuration loaded. Click "Calculate" to see updated results.</p>';
        } else {
            this.elements.plotArea.innerHTML = '<p style="color:#666;text-align:center;padding:20px;">New plane configuration loaded. Click "Calculate" to see updated results.</p>';
        }
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
            case 'open-save': this.saveConfigAdvanced(btn.dataset.type || 'plane'); break;
            case 'load': this.executeLoad(filename, type); break;
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
        };
        Object.entries(map).forEach(([k, v]) => { window[k] = v; });
    }

    // -----------------------
    // Introspección / estado
    // -----------------------
    getStatus() {
        return { initialized: true, cacheSize: this.cache.size, domFound: { messageContainer: !!this.elements.messageContainer, plotArea: !!this.elements.plotArea, airfoilForm: !!this.elements.airfoilForm, wingForm: !!this.elements.wingForm }, debug: this.config.debug };
    }

    // -----------------------
    // Inicialización
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

// Auto-inicialización con fallback
let unifiedConfigManager = null;
document.addEventListener('DOMContentLoaded', async () => {
    try {
        unifiedConfigManager = new UnifiedConfigManager();
        await unifiedConfigManager.init();
        window.unifiedConfigManager = unifiedConfigManager;
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
                    msgContainer.innerHTML = `<div class="message error">${msg}</div>`;
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
        ['saveAirfoilAdvanced', 'loadAirfoilAdvanced', 'validateAirfoilConfig', 'savePlaneAdvanced', 'loadPlaneAdvanced', 'validateConfig'].forEach(name => { window[name] = errorFn; });
    }
});
