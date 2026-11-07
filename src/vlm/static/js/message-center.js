/**
 * MessageCenter - Unified messaging system for VLMPy.
 * Uses the CSS classes: .message .success .error .info
 * Author: Javier Lendínez
 * SPDX-License-Identifier: AGPL-3.0-only
 */
class MessageCenter {
    #container;
    #timeout;
    #activeMessages = new Set();

    constructor(selector = '#message-container', defaultTimeout = 5000) {
        this.#container = document.querySelector(selector) ?? this.#createContainer(selector);
        this.#timeout = defaultTimeout;

        // Delegated click: close messages with the dismiss button.
        this.#container.addEventListener('click', e => {
            if (e.target.closest('[data-dismiss]')) {
                const message = e.target.closest('.message');
                this.#removeMessage(message);
            }
        });

        // Press Escape to clear all messages.
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') this.clear();
        });
    }

    /* Public methods. */
    info(msg, timeout)    { this.#push(msg, 'info', timeout); }
    success(msg, timeout) { this.#push(msg, 'success', timeout); }
    warning(msg, timeout) { this.#push(msg, 'warning', timeout); }
    error(msg, timeout)   { this.#push(msg, 'error', timeout); }

    clear() {
        this.#activeMessages.clear();
        this.#container.replaceChildren();
    }

    showMessage(text, type = 'info', timeout) {
        const key = String(type).toLowerCase();
        const handlers = {
            success: this.success.bind(this),
            error: this.error.bind(this),
            warning: this.warning.bind(this),
            info: this.info.bind(this),
        };
        const handler = handlers[key] || handlers.info;
        handler(text, timeout);
    }

    /* Private methods. */
    #push(text, type, timeout = this.#timeout) {
        const hash = `${type}:${text}`;
        if (this.#activeMessages.has(hash)) return;

        const message = this.#createMessage(text, type, hash);
        this.#container.appendChild(message);
        this.#activeMessages.add(hash);

        if (timeout > 0) {
            setTimeout(() => this.#removeMessage(message), timeout);
        }

        return message;
    }

    #createMessage(text, type, hash) {
        const message = document.createElement('div');
        // Apply the base 'message' class and a separate severity class.
        message.classList.add('message', String(type || 'info').toLowerCase());
        message.dataset.hash = hash;

        // Set accessibility attributes based on severity.
        if (type === 'error' || type === 'warning') {
            message.setAttribute('role', 'alert');
            message.setAttribute('aria-live', 'assertive');
        } else {
            message.setAttribute('role', 'status');
            message.setAttribute('aria-live', 'polite');
        }

        const content = document.createElement('div');
        content.className = 'message-content';
        const severity = {
            info: {icon: 'fa-info-circle', label: 'Information'},
            success: {icon: 'fa-check-circle', label: 'Success'},
            warning: {icon: 'fa-exclamation-triangle', label: 'Warning'},
            error: {icon: 'fa-times-circle', label: 'Error'},
        }[type] || {icon: 'fa-info-circle', label: 'Information'};
        const severityIcon = document.createElement('i');
        severityIcon.className = `message-icon fas ${severity.icon}`;
        severityIcon.setAttribute('aria-hidden', 'true');
        const severityLabel = document.createElement('span');
        severityLabel.className = 'message-sr-only';
        severityLabel.textContent = `${severity.label}: `;
        const textElement = document.createElement('span');
        textElement.className = 'message-text';
        textElement.textContent = String(text ?? '');
        const closeButton = document.createElement('button');
        closeButton.className = 'message-close';
        closeButton.dataset.dismiss = '';
        closeButton.setAttribute('aria-label', 'Close message');
        closeButton.type = 'button';
        const icon = document.createElement('i');
        icon.className = 'fas fa-times';
        icon.setAttribute('aria-hidden', 'true');
        closeButton.appendChild(icon);
        content.append(severityIcon, severityLabel, textElement, closeButton);
        message.appendChild(content);

        // Animate the message entrance.
        requestAnimationFrame(() => message.classList.add('message-show'));

        return message;
    }

    #removeMessage(message) {
        if (!message) return;

        const hash = message.dataset.hash;
        if (hash) this.#activeMessages.delete(hash);

        message.classList.add('message-hide');

        setTimeout(() => {
            if (message.parentNode) message.parentNode.removeChild(message);
        }, 300);
    }

    #createContainer(selector) {
        const div = document.createElement('div');
        div.id = selector.replace('#', '');
        div.className = 'message-container';
        document.body.appendChild(div);
        return div;
    }
}

/* Export and global instance. */
window.MessageCenter = MessageCenter;
window.messageCenter = new MessageCenter();

/* Route uncaught JS errors and rejected promises through notifications. */
window.addEventListener('error', e => {
    if (!e.message || e.message.startsWith('ResizeObserver')) return;
    window.messageCenter.error(`Unexpected error: ${e.message}`);
});
window.addEventListener('unhandledrejection', e => {
    const reason = e.reason;
    window.messageCenter.error(`Unexpected error: ${reason?.message ?? reason ?? 'Unknown'}`);
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MessageCenter };
}
