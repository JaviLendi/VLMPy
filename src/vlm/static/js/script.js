document.addEventListener('DOMContentLoaded', () => {
    // Cache DOM elements
    const DOM = {
        html: document.documentElement,
        sidebar: document.querySelector('.sidebar'),
        overlay: document.querySelector('.overlay'),
        menuToggle: document.getElementById('menu-toggle'),
        appearanceButtons: document.querySelectorAll('.appearance-btn'),
        plotArea: document.getElementById('plot-area'),
        messageContainer: document.getElementById('message-container'),
        calculateSpinner: document.getElementById('calculate-spinner'),
        loadingSpinner: document.getElementById('loading-spinner')
    };

    // Debug log
    console.log(
        'Initializing script.js. Sidebar:', !!DOM.sidebar,
        'Menu toggle:', !!DOM.menuToggle,
        'Appearance buttons found:', DOM.appearanceButtons.length
    );

    const sendAppearanceMode = (mode, effectiveMode) => {
        fetch('/set-appearance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ appearance_mode: mode, effective_mode: effectiveMode })
        })
            .then(resp => resp.json())
            .then(data => {
                console.log('[sendAppearanceMode] Server response:', data);
                // The backend returns something like { template: "plotly_dark" } or not.
                // We use that to toggle .dark-mode / .light-mode on <html>.
                DOM.html.classList.toggle('dark-mode', data.template === 'plotly_dark');
                DOM.html.classList.toggle('light-mode', data.template !== 'plotly_dark');

                // Update which button is “active” based on the saved mode
                updateButtonStates(mode);
            })
            .catch(err => console.error('[sendAppearanceMode] Error:', err));
    };

    const computeEffectiveMode = (mode) => {
        if (mode === 'System') {
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'Dark' : 'Light';
        }
        return mode; // "Light" or "Dark"
    };

    const updateButtonStates = (selectedMode) => {
        DOM.appearanceButtons.forEach(btn => {
            if (btn.dataset.mode === selectedMode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    };

    const wireUpButtons = () => {
        DOM.appearanceButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const chosenMode = btn.dataset.mode; // "Light" / "Dark" / "System"
                localStorage.setItem('appearance-mode', chosenMode);

                const effective = computeEffectiveMode(chosenMode);
                sendAppearanceMode(chosenMode, effective);
            });
        });
    };

    const initAppearanceMode = () => {
        const savedMode = localStorage.getItem('appearance-mode') || 'System';
        const effectiveMode = computeEffectiveMode(savedMode);

        // Send to server / toggle classes right away
        sendAppearanceMode(savedMode, effectiveMode);

        // Make sure the correct button is highlighted on load
        updateButtonStates(savedMode);

        // If the user’s OS theme toggles—and we’re in 'System'—re-send with new effective
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (localStorage.getItem('appearance-mode') === 'System') {
                const newEffective = e.matches ? 'Dark' : 'Light';
                sendAppearanceMode('System', newEffective);
            }
        });
    };

    wireUpButtons();
    initAppearanceMode();

    // Función para mostrar el spinner
    const showLoadingSpinner = () => {
        if (DOM.loadingSpinner) {
            DOM.loadingSpinner.style.display = 'block';
            console.log('Spinner shown');
        } else {
            console.error('Loading spinner element not found');
        }
    };

    // Función para ocultar el spinner
    const hideLoadingSpinner = () => {
        if (DOM.loadingSpinner) {
            DOM.loadingSpinner.style.display = 'none';
            console.log('Spinner hidden');
        }
    };

    // Evento para el enlace "Results"
    const resultsLink = document.querySelector('a[href="/results"]');
    if (resultsLink) {
        resultsLink.addEventListener('click', (event) => {
            showLoadingSpinner();
            // Optional: Delay navigation to make spinner visible
            // event.preventDefault();
            // setTimeout(() => {
            //     window.location.href = resultsLink.href;
            // }, 500);
        });
    }

    // Ocultar el spinner cuando la página esté completamente cargada
    window.addEventListener('load', () => {
        hideLoadingSpinner();
    });


    // Plot data function
    window.plotData = (endpoint, filename) => {
        if (!DOM.plotArea) {
            console.error('[plotData] Plot area not found.');
            return;
        }

        const nSectionInput = document.getElementById('n_section');
        const data = {
            n_section: nSectionInput ? nSectionInput.value : 1
        };

        fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(data => {
                if (!DOM.messageContainer) return;

                DOM.messageContainer.innerHTML = '';
                if (data.status === 'error') {
                    const msgError = document.createElement('p');
                    msgError.classList.add('message', 'error');
                    msgError.textContent = data.message;
                    DOM.messageContainer.appendChild(msgError);
                    return;
                }

                const plotData = JSON.parse(data.plot);
                const config = {
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d'],
                    displayModeBar: true,
                    toImageButtonOptions: { filename, format: 'svg' }
                };

                Plotly.newPlot(DOM.plotArea, plotData.data, plotData.layout, config);
            })
            .catch(error => console.error('[plotData] Error:', error));
    };

    // Toggle sidebar
    const toggleSidebar = () => {
        if (!DOM.menuToggle || !DOM.sidebar || !DOM.overlay) {
            console.error('Sidebar, overlay, or menu-toggle elements missing');
            return;
        }

        DOM.menuToggle.addEventListener('click', () => {
            const isOpen = DOM.sidebar.getAttribute('aria-hidden') === 'true';
            DOM.sidebar.setAttribute('aria-hidden', !isOpen);
            DOM.overlay.setAttribute('aria-hidden', !isOpen);
            DOM.menuToggle.setAttribute('aria-expanded', isOpen);
            console.log('Sidebar toggled. Open:', isOpen);
        });

        DOM.overlay.addEventListener('click', () => {
            DOM.sidebar.setAttribute('aria-hidden', 'true');
            DOM.overlay.setAttribute('aria-hidden', 'true');
            DOM.menuToggle.setAttribute('aria-expanded', 'false');
            console.log('Sidebar closed via overlay');
        });
    };

    // Toggle section or flap
    const toggleElement = (element, button, hiddenInput, isFlap = false) => {
        if (!button || !element || !hiddenInput) {
            console.error(`Toggle${isFlap ? 'Flap' : 'Section'}: Element, button, or hidden input not found`);
            return;
        }

        const isHidden = element.style.display === 'none' || element.style.display === '';
        element.style.display = isHidden ? 'block' : 'none';
        button.classList.toggle('is-toggled', isHidden);
        element.style.transition = `opacity ${isHidden ? 0.2 : 0.3}s ease`;
        hiddenInput.value = isHidden ? '1' : '0';

        setTimeout(() => {
            element.style.opacity = isHidden ? '1' : '0';
            if (!isFlap) element.classList.toggle('visible', isHidden);
            button.disabled = false;
            console.log(`${isFlap ? 'Flap' : 'Section'} ${isHidden ? 'shown' : 'hidden'}:`, button.id || element.id);
        }, isHidden ? 100 : 300);

        button.setAttribute('aria-expanded', isHidden);
        if (isFlap) button.setAttribute('aria-pressed', isHidden);
    };

    window.toggleFlap = (button) => {
        toggleElement(button.nextElementSibling, button, button.previousElementSibling, true);
    };

    window.toggleSection = (sectionId, buttonSelector) => {
        const section = document.getElementById(sectionId);
        const button = document.querySelector(buttonSelector);
        toggleElement(section, button, button.previousElementSibling);
    };

    // Add wing section
    window.addSection = () => {
        const sections = document.getElementById('sections');
        if (!sections) {
            console.error('Sections container not found');
            return;
        }

        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'section';
        sectionDiv.style.opacity = '0';
        const sectionNumber = sections.children.length + 1;

        const defaultSection = (typeof defaultPlane !== 'undefined' && defaultPlane.wing_sections && defaultPlane.wing_sections.length > 0)
            ? defaultPlane.wing_sections[0]
            : {};

        const defaults = {
            chord_root: defaultSection.chord_root || 1.0,
            chord_tip: defaultSection.chord_tip || 0.8,
            span_fraction: defaultSection.span_fraction || 8.0,
            sweep: defaultSection.sweep || 15,
            dihedral: defaultSection.dihedral || 5,
            naca_root: defaultSection.NACA_root || '3215',
            naca_tip: defaultSection.NACA_tip || '1310',
            flap_start: defaultSection.flap_start || 0.2,
            flap_end: defaultSection.flap_end || 0.8,
            flap_hinge_chord: defaultSection.flap_hinge_chord || 0.25,
            deflection_angle: defaultSection.deflection_angle || 20,
            deflection_type: defaultSection.deflection_type || 'symmetrical'

        };

        sectionDiv.innerHTML = `
            <h4>Section ${sectionNumber}</h4>
            <label>Chord Root (m):</label>
            <input type="number" name="chord_root[]" step="0.1" required aria-required="true" value="${defaults.chord_root}" placeholder="e.g., 1.5">
            <label>Chord Tip (m):</label>
            <input type="number" name="chord_tip[]" step="0.1" required aria-required="true" value="${defaults.chord_root}" placeholder="e.g., 0.8">
            <label>Span Fraction (m):</label>
            <input type="number" name="span_fraction[]" step="any" required aria-required="true" value="${defaults.span_fraction}" placeholder="e.g., 0.25">
            <label>Sweep Angle (deg):</label>
            <input type="number" name="sweep[]" step="any" required aria-required="true" value="${defaults.sweep}" placeholder="e.g., 15">
            <label>Dihedral Angle (deg):</label>
            <input type="number" name="dihedral[]" step="any" required aria-required="true" value="${defaults.dihedral}" placeholder="e.g., 5">
            <label>NACA Root:</label>
            <input type="text" name="naca_root[]" required aria-required="true" value="${defaults.naca_root}" placeholder="e.g., 2412">
            <label>NACA Tip:</label>
            <input type="text" name="naca_tip[]" required aria-required="true" value="${defaults.naca_tip}" placeholder="e.g., 0012">
            <input type="hidden" name="flap_toggled[]" value='0'>
            <button type="button" class="toggle-button" id="aileron-${sectionNumber}" aria-label="Toggle Aileron">Toggle Aileron</button>
            <div class="flap-params" style="display: none;">
                <label>Flap Start (fraction):</label>
                <input type="number" name="flap_start[]" step="any" value="${defaults.flap_start}" placeholder="e.g., 0.2">
                <label>Flap End (fraction):</label>
                <input type="number" name="flap_end[]" step="any" value="${defaults.flap_end}" placeholder="e.g., 0.8">
                <label>Flap Hinge Chord (fraction):</label>
                <input type="number" name="flap_hinge_chord[]" step="any" value="${defaults.flap_hinge_chord}" placeholder="e.g., 0.25">
                <label>Deflection Angle (deg):</label>
                <input type="number" name="deflection_angle[]" step="any" value="${defaults.deflection_angle}" placeholder="e.g., 20">
                <label for="deflection-type">Deflection Type:</label>
                <select name="deflection_type[]">
                    <option value="symmetrical">Symmetrical</option>
                    <option value="antisymmetrical">Antisymmetrical</option>
                </select>
            </div>
        `;

        sections.appendChild(sectionDiv);
        setTimeout(() => {
            sectionDiv.style.transition = 'opacity 0.3s';
            sectionDiv.style.opacity = '1';
        }, 10);

        const newButton = sectionDiv.querySelector(`#aileron-${sectionNumber}`);
        if (newButton) {
            newButton.addEventListener('click', () => toggleFlap(newButton));
        } else {
            console.error('Aileron button not found for section', sectionNumber);
        }
    };

    // Remove wing section
    window.removeSection = () => {
        const sections = document.getElementById('sections');
        if (!sections || sections.children.length <= 1) {
            console.log('Cannot remove: Sections container missing or only one section remains');
            return;
        }

        const lastSection = sections.lastChild;
        lastSection.style.transition = 'opacity 0.3s ease';
        lastSection.style.opacity = '0';
        setTimeout(() => {
            sections.removeChild(lastSection);
            console.log('Section removed');
        }, 300);
    };

    // Handle form submission
    const handleFormSubmission = (form, endpoint, formName) => {
        if (!form) {
            console.error(`${formName} form not found`);
            return Promise.reject(new Error(`${formName} form not found`));
        }
        if (!form.checkValidity()) {
            form.reportValidity();
            console.log('Form validation failed');
            return Promise.reject(new Error('Form validation failed'));
        }

        return fetch(endpoint, {
            method: 'POST',
            body: new FormData(form)
        })
            .then(response => response.json())
            .then(data => {
                if (!DOM.messageContainer) {
                    console.error('Message container not found');
                    return;
                }

                DOM.messageContainer.innerHTML = '';
                const messageElement = document.createElement('p');
                messageElement.classList.add('message', data.status);
                messageElement.textContent = data.message;
                DOM.messageContainer.appendChild(messageElement);
                console.log(`${formName} result:`, data.status, data.message);
            })
            .catch(error => {
                console.error(`Error during ${formName.toLowerCase()} calculation:`, error);
                if (DOM.messageContainer && DOM.plotArea) {
                    DOM.plotArea.innerHTML = `<p style="color: red;">Error: Failed to process request</p>`;
                }
                throw error;
            });
    };

    window.Calculate = () => {
        const wingForm = document.getElementById('wing-form');
        handleFormSubmission(wingForm, '/wing', 'Wing');
    };

    window.CalculateAngles = () => {
        const analisisForm = document.getElementById('analisis-form');
        if (DOM.calculateSpinner) DOM.calculateSpinner.style.display = 'inline-block';
        handleFormSubmission(analisisForm, '/angles', 'Analisis')
            .finally(() => {
                if (DOM.calculateSpinner) DOM.calculateSpinner.style.display = 'none';
            });
    };

    // Initialize flap-params visibility
    document.querySelectorAll('.flap-params').forEach(params => {
        params.style.opacity = '0';
        params.style.display = 'none';
    });

    // Bind toggle buttons
    document.querySelectorAll('.toggle-button').forEach(button => {
        button.addEventListener('click', () => {
            button.classList.toggle('is-toggled');
            button.setAttribute('aria-pressed', button.classList.contains('is-toggled'));
            if (button.id?.startsWith('aileron-')) {
                toggleFlap(button);
            }
        });
    });

    // Initialize components
    initAppearanceMode();
    toggleSidebar();
});