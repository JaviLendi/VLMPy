(() => {
    const mode = document.getElementById('fuselage-mode');
    if (!mode) return;
    const stations = document.getElementById('fuselage-stations');
    const file = document.getElementById('fuselage-file');
    const units = document.getElementById('fuselage-units');
    const configField = document.getElementById('fuselage-config');
    const status = document.getElementById('fuselage-status');
    const report = document.getElementById('fuselage-report');
    const component = document.getElementById('fuselage-component');
    const scale = document.getElementById('fuselage-scale');
    const addStation = document.getElementById('fuselage-add-station');
    const axes = ['x', 'y', 'z'];
    const placement = kind => axes.map(axis => document.getElementById(`fuselage-${kind}-${axis}`));
    const defaultStations = [
        {x: -7, y_radius: 0, z_radius: 0},
        {x: -5, y_radius: 0.6, z_radius: 0.6},
        {x: -2, y_radius: 0.6, z_radius: 0.6},
        {x: 0, y_radius: 0, z_radius: 0}
    ];
    let imported = null;
    let sequence = 0;

    function stationRows() {
        return [...stations.querySelectorAll('.fuselage-station')];
    }

    function updateStationLabels() {
        const rows = stationRows();
        rows.forEach((row, index) => {
            const name = index === 0 ? 'Tail' : index === rows.length - 1 ? 'Nose' : `Station ${index}`;
            row.querySelector('.fuselage-station-name').textContent = name;
            row.querySelectorAll('input').forEach(input => input.setAttribute('aria-label', `${name} ${input.dataset.label}`));
            const remove = row.querySelector('.fuselage-remove-station');
            remove.hidden = index === 0 || index === rows.length - 1;
            remove.disabled = rows.length <= 3;
            remove.setAttribute('aria-label', `Remove ${name}`);
            remove.title = `Remove ${name}`;
            const insert = row.querySelector('.fuselage-insert-station');
            insert.hidden = index === rows.length - 1;
            insert.disabled = rows.length >= 500;
            insert.setAttribute('aria-label', `Insert station after ${name}`);
            insert.title = `Insert station after ${name}`;
        });
        addStation.disabled = rows.length >= 500;
    }

    function createStation(station) {
        const row = document.createElement('div');
        row.className = 'fuselage-station';
        const header = document.createElement('div');
        header.className = 'fuselage-station-header';
        const name = document.createElement('span');
        name.className = 'fuselage-station-name';
        const actions = document.createElement('div');
        actions.className = 'fuselage-station-actions';
        const insert = document.createElement('button');
        insert.type = 'button';
        insert.className = 'fuselage-insert-station';
        insert.innerHTML = '<span class="material-icons" aria-hidden="true">add</span>';
        insert.addEventListener('click', () => insertStationBefore(row.nextElementSibling));
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'fuselage-remove-station';
        remove.innerHTML = '<span class="material-icons" aria-hidden="true">delete_outline</span>';
        remove.addEventListener('click', () => {
            row.remove();
            updateStationLabels();
            refresh();
        });
        actions.append(insert, remove);
        header.append(name, actions);
        const fields = document.createElement('div');
        fields.className = 'fuselage-station-fields';
        [['x', 'X position'], ['y_radius', 'Y radius'], ['z_radius', 'Z radius']].forEach(([key, label]) => {
            const field = document.createElement('label');
            field.textContent = label;
            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.value = station[key] ?? '';
            input.dataset.field = key;
            input.dataset.label = label;
            input.addEventListener('input', refresh);
            field.append(input);
            fields.append(field);
        });
        row.append(header, fields);
        return row;
    }

    function populateStations(values) {
        stations.replaceChildren(...values.map(createStation));
        updateStationLabels();
    }

    function invalid(message, input) {
        const error = new Error(message);
        error.input = input;
        return error;
    }

    function configuration() {
        if (mode.value === 'none') return null;
        let body;
        if (mode.value === 'stations') {
            const rows = stationRows();
            const values = rows.map((row, index) => {
                const inputs = [...row.querySelectorAll('input')];
                const empty = inputs.find(input => input.value === '' || !Number.isFinite(Number(input.value)));
                if (empty) throw invalid(`Enter a finite value for ${empty.getAttribute('aria-label')}.`, empty);
                const station = Object.fromEntries(inputs.map(input => [input.dataset.field, Number(input.value)]));
                const radius = inputs.find(input => input.dataset.field !== 'x' &&
                    (index === 0 || index === rows.length - 1 ? Number(input.value) !== 0 : Number(input.value) <= 0));
                if (radius) throw invalid(index === 0 || index === rows.length - 1
                    ? 'Nose and tail radii must be zero.' : 'Interior station radii must be positive.', radius);
                return station;
            });
            for (let index = 1; index < values.length; index++) {
                if (values[index].x <= values[index - 1].x) {
                    throw invalid('Station positions must increase from tail to nose.', rows[index].querySelector('[data-field="x"]'));
                }
            }
            body = {type: 'stations', stations: values};
        } else {
            if (!imported) throw new Error('Import an STL file first.');
            if (scale.value === '' || !Number.isFinite(Number(scale.value)) || Number(scale.value) <= 0) {
                throw invalid('STL scale must be a positive finite number.', scale);
            }
            body = {...imported, selected_component: component.value === '' ? null : Number(component.value), scale: Number(scale.value)};
        }
        const placementInputs = [...placement('translate'), ...placement('rotate')];
        const invalidPlacement = placementInputs.find(input => input.value === '' || !Number.isFinite(Number(input.value)));
        if (invalidPlacement) throw invalid('Placement values must be finite numbers.', invalidPlacement);
        const translation = placement('translate').map(input => Number(input.value));
        const rotation = placement('rotate').map(input => Number(input.value));
        return {...body, translation, rotation_deg: rotation};
    }

    async function refresh() {
        const requestId = ++sequence;
        document.getElementById('fuselage-stations-group').hidden = mode.value !== 'stations';
        document.getElementById('fuselage-import-group').hidden = mode.value !== 'stl';
        document.getElementById('fuselage-placement').hidden = mode.value === 'none';
        document.getElementById('fuselage-scale-group').hidden = mode.value !== 'stl';
        stations.querySelectorAll('input').forEach(input => input.setCustomValidity(''));
        [...placement('translate'), ...placement('rotate')].forEach(input => input.setCustomValidity(''));
        scale.setCustomValidity('');
        file.setCustomValidity('');
        try {
            const body = configuration();
            configField.value = body ? JSON.stringify(body) : '';
            if (!body) {
                window.fuselagePreview = null;
                status.textContent = '';
                window.renderWingPreview?.();
                return;
            }
            const response = await window.vlmpyCore.request('/api/fuselage/preview', {method: 'POST', body});
            if (requestId !== sequence) return;
            window.fuselagePreview = response;
            status.textContent = 'Geometry only; excluded from aerodynamic loads.';
            window.renderWingPreview?.();
        } catch (error) {
            if (requestId !== sequence) return;
            configField.value = '';
            window.fuselagePreview = null;
            status.textContent = error.message;
            const target = error.input || (mode.value === 'stations' ? stations.querySelector('input') : file);
            target?.setCustomValidity(error.message);
            window.renderWingPreview?.();
        }
    }

    document.getElementById('fuselage-import').addEventListener('click', async () => {
        if (!file.files.length || !units.value) {
            status.textContent = 'Select an STL file and its units.';
            return;
        }
        const form = new FormData();
        form.append('file', file.files[0]);
        form.append('units', units.value);
        form.append('source_tool', document.getElementById('fuselage-source-tool').value);
        form.append('coordinate_system', document.getElementById('fuselage-source-frame').value);
        form.append('export_settings', document.getElementById('fuselage-export-settings').value);
        try {
            const response = await window.vlmpyCore.request('/api/fuselage/import', {method: 'POST', body: form});
            imported = response.fuselage;
            const info = response.report;
            component.replaceChildren(new Option('All components', ''));
            info.components.forEach((item, index) => component.add(new Option(`Component ${index + 1}: ${item.triangle_count} triangles, bounds ${JSON.stringify(item.bounds_m)} m`, String(index))));
            report.textContent = `${info.format.toUpperCase()} | ${info.triangle_count} triangles | ${info.connected_components} components | ${info.duplicate_vertices} shared vertex occurrences | ${info.boundary_edges} boundary edges | ${info.nonmanifold_edges} nonmanifold edges | ${info.inconsistent_edges} inconsistent edges | ${info.missing_normals} missing and ${info.opposed_normals} opposed normals | bounds (m): ${JSON.stringify(info.bounds_m)}`;
            await refresh();
        } catch (error) {
            imported = null;
            report.textContent = '';
            await refresh();
            status.textContent = error.message;
        }
    });
    function insertStationBefore(next) {
        const previous = next?.previousElementSibling;
        if (!previous || stationRows().length >= 500) return;
        const value = (row, key) => Number(row.querySelector(`[data-field="${key}"]`).value);
        const left = value(previous, 'x');
        const right = value(next, 'x');
        const station = {
            x: Number.isFinite(left) && Number.isFinite(right) && left < right ? left + (right - left) / 2 : '',
            y_radius: value(previous, 'y_radius') / 2 + value(next, 'y_radius') / 2,
            z_radius: value(previous, 'z_radius') / 2 + value(next, 'z_radius') / 2
        };
        const row = createStation(station);
        stations.insertBefore(row, next);
        updateStationLabels();
        refresh();
        row.querySelector('input').focus();
    }
    addStation.addEventListener('click', () => insertStationBefore(stations.lastElementChild));
    [mode, component, scale, ...placement('translate'), ...placement('rotate')].forEach(input => input.addEventListener('input', refresh));
    [file, units].forEach(input => input.addEventListener('change', () => {
        imported = null;
        report.textContent = '';
        component.replaceChildren(new Option('All components', ''));
        refresh();
    }));
    window.addEventListener('vlmpy:configuration-loaded', event => {
        const body = event.detail?.fuselage;
        imported = body?.type === 'stl' ? body : null;
        component.replaceChildren(new Option('All components', ''));
        if (imported?.component_ids) {
            [...new Set(imported.component_ids)].sort((first, second) => first - second).forEach(index => component.add(new Option(`Component ${index + 1}`, String(index))));
            component.value = imported.selected_component == null ? '' : String(imported.selected_component);
        }
        document.getElementById('fuselage-source-tool').value = imported?.source?.source_tool || '';
        document.getElementById('fuselage-source-frame').value = imported?.source?.coordinate_system || '';
        document.getElementById('fuselage-export-settings').value = imported?.source?.export_settings || '';
        mode.value = body?.type || 'none';
        scale.value = body?.scale ?? 1;
        populateStations(body?.type === 'stations' ? body.stations : defaultStations);
        placement('translate').forEach((input, index) => { input.value = body?.translation?.[index] ?? 0; });
        placement('rotate').forEach((input, index) => { input.value = body?.rotation_deg?.[index] ?? 0; });
        report.textContent = imported ? `Loaded: ${imported.source?.filename || 'STL'}` : '';
        refresh();
    });
    populateStations(defaultStations);
})();