/*
 * Author: Javier Lendínez
 * SPDX-License-Identifier: AGPL-3.0-only
 */
(() => {
    const activeRequests = new Map();

    async function request(endpoint, options = {}) {
        const method = (options.method || 'GET').toUpperCase();
        const controller = new AbortController();
        const headers = new Headers(options.headers || {});
        let body = options.body;

        if (body && !(body instanceof FormData) && typeof body !== 'string') {
            headers.set('Content-Type', 'application/json');
            body = JSON.stringify(body);
        }
        if (!headers.has('Accept')) headers.set('Accept', 'application/json');

        const previous = activeRequests.get(endpoint);
        previous?.abort();
        activeRequests.set(endpoint, controller);

        try {
            const response = await fetch(endpoint, {method, headers, body, signal: controller.signal});
            const text = await response.text();
            let data = {};
            try { data = text ? JSON.parse(text) : {}; } catch { data = {message: text}; }
            if (!response.ok || data.status === 'error') {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            return data;
        } finally {
            if (activeRequests.get(endpoint) === controller) activeRequests.delete(endpoint);
        }
    }

    function setBusy(element, busy, busyText = 'Loading...') {
        if (!element) return;
        if (busy) {
            element.dataset.defaultText = element.textContent;
            element.disabled = true;
            element.textContent = busyText;
        } else {
            element.disabled = false;
            if (element.dataset.defaultText) element.textContent = element.dataset.defaultText;
        }
    }

    window.vlmpyCore = {request, setBusy};
})();
