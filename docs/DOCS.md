# VLMPy: Vortex Lattice Method

<div align="center">

![VLMPy Logo](https://raw.githubusercontent.com/JaviLendi/VLMPy/main/docs/img/icon.png) <!-- Cambia la URL si tienes un logo propio -->

**Autor:** Javier Lendínez Castillo  
**Tutor:** Héctor Gómez Cedenilla  
**Fecha:** Junio de 2025

</div>

---

## 📑 Tabla de Contenidos

- [VLMPy: Vortex Lattice Method](#vlmpy-vortex-lattice-method)
  - [📑 Tabla de Contenidos](#-tabla-de-contenidos)
  - [🚀 Introducción](#-introducción)
  - [✨ Características Principales](#-características-principales)
  - [⚙️ Instalación y Configuración](#️-instalación-y-configuración)
    - [🛠️ Prerrequisitos](#️-prerrequisitos)
    - [📥 Instrucciones Paso a Paso](#-instrucciones-paso-a-paso)
      - [💻 Usando el Ejecutable (Solo Windows 11)](#-usando-el-ejecutable-solo-windows-11)
  - [🧑‍💻 Ejemplos de Uso](#-ejemplos-de-uso)
    - [✈️ Ejemplo: Análisis de un Ala Rectangular](#️-ejemplo-análisis-de-un-ala-rectangular)
  - [⚠️ Limitaciones](#️-limitaciones)
  - [📄 Licencia](#-licencia)

---

## 🚀 Introducción

> **VLMPy** es una herramienta de código abierto para el análisis aerodinámico de alas mediante el método Vortex Lattice (VLM).  
> Ideal para ingenieros, estudiantes y entusiastas de la aeronáutica que buscan una solución intuitiva y potente para modelar la sustentación y resistencia en alas.

El método VLM es ampliamente utilizado en dinámica de fluidos computacional (CFD) para predecir el comportamiento aerodinámico de alas en condiciones de flujo subsónico e invíscido. VLMPy implementa este método en Python, facilitando el análisis y la visualización de resultados.

<div align="center">

![VLM Schematic](https://raw.githubusercontent.com/JaviLendi/VLMPy/main/docs/img/vlm_schematic.png) <!-- Cambia la URL si tienes un esquema propio -->

</div>

Este proyecto surge como parte del Trabajo de Fin de Grado en Ingeniería Aeroespacial, conectando la teoría aerodinámica con aplicaciones prácticas de ingeniería.

---

## ✨ Características Principales

- **Cálculo de Coeficientes Aerodinámicos:**  
    Obtén $C_L$, $C_D$ y momentos para diversas configuraciones de alas.
- **Análisis de Fuerzas y Momentos** *(próximamente)*  
    Calcula sustentación, resistencia y momentos para el ala completa o secciones.
- **Visualización Interactiva:**  
    Gráficos 2D y 3D con Plotly para geometría, presión y fuerzas.
- **Geometría de Ala Personalizable:**  
    Soporta flecha, diedro, conicidad y superficies de control (flaps, alerones).
- **Condiciones de Vuelo Múltiples:**  
    Analiza diferentes ángulos de ataque, resbalamiento y velocidades.
- **Soporte para Perfiles NACA:**  
    Genera y analiza perfiles estándar NACA.
- **Arquitectura Modular:**  
    Extiende o modifica el código fácilmente para investigación avanzada.

---

## ⚙️ Instalación y Configuración

### 🛠️ Prerrequisitos

- **Python 3.8+**
- **Git**
- **Navegador Web:** Chrome 90+, Firefox 85+ o equivalente

### 📥 Instrucciones Paso a Paso

1. **Clona el repositorio:**
        ```bash
        git clone https://github.com/JaviLendi/VLMPy.git
        cd VLMPy
        ```

2. **Instala las dependencias:**
        ```bash
        pip install -r requirements.txt
        ```
        > 💡 *Si tienes problemas, actualiza pip con `pip install --upgrade pip`.*

3. **Ejecuta la aplicación:**
        ```bash
        python app.py
        ```
        O ejecuta `app.exe` (solo Windows 11).

4. **Accede a la interfaz:**  
     Abre [http://localhost:5000](http://localhost:5000) en tu navegador.

#### 💻 Usando el Ejecutable (Solo Windows 11)

Descarga `app.exe` desde la página de lanzamientos y haz doble clic para iniciar la aplicación.

---

## 🧑‍💻 Ejemplos de Uso

A continuación, un flujo típico para analizar un ala rectangular:

### ✈️ Ejemplo: Análisis de un Ala Rectangular

1. **Definir el Perfil Aerodinámico**
        - Ve a "Definición NACA".
        - Ingresa el código NACA (ej. 0012).
        - Establece cuerda, ángulo de ataque y velocidad.
        - Haz clic en "Crear Gráfico".

2. **Configurar el Ala**
        - Ve a "Definición del Avión".
        - Parámetros:  
            - Cuerda raíz: 1 m  
            - Cuerda punta: 1 m  
            - Envergadura: 10 m  
            - Flecha/diedro: 0°
        - Selecciona el perfil NACA.
        - (Opcional) Añade flaps o superficies de control.

3. **Condiciones de Vuelo**
        - Velocidad: 50 m/s
        - Densidad: 1.225 kg/m³
        - Ángulo de ataque: 5°
        - Resbalamiento: 0°

4. **Discretización**
        - Paneles envergadura (n): 20
        - Paneles cuerda (m): 10

5. **Ejecutar el Cálculo**
        - Haz clic en "Calcular".
        - Visualiza la geometría y discretización en 2D/3D.

6. **Analizar Resultados**
        - Ve a "Resultados" para iniciar el análisis VLM.
        - Observa $C_L$, $C_D$ y distribuciones de presión.
        - Exporta resultados a CSV o JSON.

> Para ejemplos avanzados (alas con flecha, conicidad, superficies de control), consulta el manual de usuario *(en desarrollo)*.

---

## ⚠️ Limitaciones

- **Suposiciones de flujo:** invíscido e incompresible (no apto para transónico/supersónico o efectos viscosos).
- **Geometría:** optimizado para alas delgadas; no modela alas gruesas o fuselajes.
- **Recursos:** muchos paneles aumentan el tiempo de cálculo.
- **Sistema Operativo:** el ejecutable solo probado en Windows 11 (debería funcionar en Windows 10).

---

## 📄 Licencia

VLMPy se distribuye bajo la **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
Consulta el archivo `LICENSE` para más detalles.

---

> ℹ️ **Nota:** Para detalles académicos y derivaciones matemáticas, revisa `docs/TFG.pdf` (próximamente) y los apéndices de código fuente.

