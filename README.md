# Tarea 2 - Inteligencia Artificial 2025

Este proyecto implementa las tres partes de la Tarea 2:
1. Algoritmo K-means
2. Algoritmo DBScan  
3. Aprendizaje por refuerzo (Q-learning y SARSA)

## Configuración del entorno

### 1. Crear y activar entorno virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # macOS/Linux
# o
venv\Scripts\activate     # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecución

### Parte 1 y 2: K-means y DBScan

```bash
# Asegúrate de tener el entorno virtual activado
source venv/bin/activate

# Ejecutar algoritmos de clustering
python3 kmeans_dbscan.py
```

Este script ejecutará:
- K-means con 3 iteraciones usando centroides iniciales A, D, G
- DBScan con eps=2 y minPts=2
- DBScan con eps=10 y minPts=2
- Generará visualizaciones en grillas 10x10

### Parte 3: Aprendizaje por refuerzo

#### 1. Compilar el programa C++

```bash
g++ -o tutorial tutorial.cpp
```

#### 2. Ejecutar experimentos

Modificar las variables en `tutorial.cpp` para cada experimento:

**Q-learning Ambiente 1:**
```cpp
int algorithm = 1; // Q-learning
int environment = 1; // Grid 3x4
```

**Q-learning Ambiente 2:**
```cpp
int algorithm = 1; // Q-learning  
int environment = 2; // Cliff walking
```

**SARSA Ambiente 1:**
```cpp
int algorithm = 2; // SARSA
int environment = 1; // Grid 3x4
```

**SARSA Ambiente 2:**
```cpp
int algorithm = 2; // SARSA
int environment = 2; // Cliff walking
```

#### 3. Ejecutar cada configuración

```bash
# Recompilar después de cada cambio
g++ -o tutorial tutorial.cpp

# Ejecutar
./tutorial

# Renombrar el archivo de salida después de cada ejecución
mv Rewards.txt Rewards_Qlearning_Env1.txt    # Para Q-learning Ambiente 1
mv Rewards.txt Rewards_Qlearning_Env2.txt    # Para Q-learning Ambiente 2  
mv Rewards.txt Rewards_SARSA_Env1.txt        # Para SARSA Ambiente 1
mv Rewards.txt Rewards_SARSA_Env2.txt        # Para SARSA Ambiente 2
```

#### 4. Generar análisis y visualizaciones

```bash
# Una vez que tengas los 4 archivos de resultados
python3 analyze_results.py
```

## Configuraciones implementadas

### Parámetros de aprendizaje por refuerzo:
- **Tasa de aprendizaje (α):** 0.1
- **Factor de descuento (γ):** 0.99  
- **Tasa de exploración (ε):** 0.05
- **Episodios:** 3000
- **Selección de acción:** Epsilon-greedy
- **Acciones estocásticas:** 80% intención, 10% derecha, 10% izquierda

### Ambientes:
1. **Ambiente 1:** Grid 3x4 con obstáculo y estados terminales
2. **Ambiente 2:** Cliff Walking 4x12

### Algoritmos implementados:
- **Q-learning:** Off-policy, usa max Q(s',a') para actualización
- **SARSA:** On-policy, usa Q(s',a') de la acción seleccionada

## Archivos del proyecto

- `kmeans_dbscan.py`: Implementación de K-means y DBScan
- `tutorial.cpp`: Código de aprendizaje por refuerzo completado
- `analyze_results.py`: Script para generar curvas de aprendizaje y análisis
- `requirements.txt`: Dependencias de Python
- `CLAUDE.md`: Documentación para Claude Code

## Resultados esperados

El proyecto generará:
- Visualizaciones de clusters para K-means y DBScan
- Archivos CSV con rewards por episodio para cada algoritmo/ambiente
- Curvas de aprendizaje comparativas
- Análisis estadístico del desempeño de los algoritmos