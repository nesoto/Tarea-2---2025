import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import math

# Datos originales
datos = {
    'A': (2, 10), 'B': (2, 5), 'C': (8, 4), 'D': (5, 8),
    'E': (7, 5), 'F': (6, 4), 'G': (1, 2), 'H': (4, 9)
}

# Convertir a array numpy
puntos = np.array(list(datos.values()))
nombres = list(datos.keys())

# Matriz de distancias proporcionada
distancias = np.array([
    [0, 5, 8.54, 3.61, 7.07, 7.21, 8.06, 2.24],
    [5, 0, 6.08, 4.24, 5, 4.12, 3.16, 4.47],
    [8.54, 6.08, 0, 5, 1.41, 2, 7.28, 6.4],
    [3.61, 4.24, 5, 0, 3.61, 4.12, 7.21, 1.41],
    [7.07, 5, 1.41, 3.61, 0, 1.41, 6.71, 5],
    [7.21, 4.12, 2, 4.12, 1.41, 0, 5.39, 5.39],
    [8.06, 3.16, 7.28, 7.21, 6.71, 5.39, 0, 7.62],
    [2.24, 4.47, 6.4, 1.41, 5, 5.39, 7.62, 0]
])

def calcular_distancia_euclidiana(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def kmeans_manual(puntos, centroides_iniciales, num_iteraciones=3):
    """
    Implementación manual de K-means
    """
    centroides = np.array(centroides_iniciales)
    historial_clusters = []
    historial_centroides = []
    
    for iteracion in range(num_iteraciones):
        print(f"\n=== ITERACIÓN {iteracion + 1} ===")
        
        # Asignar puntos a clusters
        clusters = [[] for _ in range(len(centroides))]
        asignaciones = []
        
        for i, punto in enumerate(puntos):
            distancias_punto = []
            for centroide in centroides:
                dist = calcular_distancia_euclidiana(punto, centroide)
                distancias_punto.append(dist)
            
            cluster_asignado = np.argmin(distancias_punto)
            clusters[cluster_asignado].append(i)
            asignaciones.append(cluster_asignado)
        
        # Mostrar clusters
        print("Clusters:")
        for i, cluster in enumerate(clusters):
            puntos_cluster = [nombres[j] for j in cluster]
            print(f"Cluster {i+1}: {puntos_cluster}")
        
        historial_clusters.append(clusters)
        
        # Calcular nuevos centroides
        nuevos_centroides = []
        for cluster in clusters:
            if len(cluster) > 0:
                puntos_cluster = puntos[cluster]
                nuevo_centroide = np.mean(puntos_cluster, axis=0)
                nuevos_centroides.append(nuevo_centroide)
            else:
                nuevos_centroides.append(centroides[len(nuevos_centroides)])
        
        centroides = np.array(nuevos_centroides)
        
        print("Nuevos centroides:")
        for i, centroide in enumerate(centroides):
            print(f"Centroide {i+1}: ({centroide[0]:.2f}, {centroide[1]:.2f})")
        
        historial_centroides.append(centroides.copy())
        
        # Visualizar
        visualizar_clusters(puntos, asignaciones, centroides, f"K-means Iteración {iteracion + 1}")
    
    return historial_clusters, historial_centroides

def visualizar_clusters(puntos, asignaciones, centroides, titulo):
    """
    Visualiza los clusters en una grilla 10x10
    """
    plt.figure(figsize=(10, 10))
    colores = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Dibujar puntos
    for i, punto in enumerate(puntos):
        color = colores[asignaciones[i]]
        plt.scatter(punto[0], punto[1], c=color, s=100, alpha=0.7)
        plt.annotate(nombres[i], (punto[0], punto[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold')
    
    # Dibujar centroides
    for i, centroide in enumerate(centroides):
        plt.scatter(centroide[0], centroide[1], c=colores[i], s=200, marker='x', linewidths=3)
        plt.annotate(f'C{i+1}', (centroide[0], centroide[1]), xytext=(5, -15), 
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def dbscan_manual(puntos, eps, min_pts):
    """
    Implementación manual de DBSCAN
    """
    n = len(puntos)
    visitado = [False] * n
    cluster_id = [-1] * n  # -1 significa ruido
    cluster_actual = 0
    
    def obtener_vecinos(punto_idx):
        vecinos = []
        for i in range(n):
            if i != punto_idx:
                dist = calcular_distancia_euclidiana(puntos[punto_idx], puntos[i])
                if dist <= eps:
                    vecinos.append(i)
        return vecinos
    
    for i in range(n):
        if visitado[i]:
            continue
            
        visitado[i] = True
        vecinos = obtener_vecinos(i)
        
        if len(vecinos) < min_pts:
            cluster_id[i] = -1  # Ruido
        else:
            cluster_id[i] = cluster_actual
            
            # Expandir cluster
            j = 0
            while j < len(vecinos):
                vecino = vecinos[j]
                if not visitado[vecino]:
                    visitado[vecino] = True
                    vecinos_del_vecino = obtener_vecinos(vecino)
                    if len(vecinos_del_vecino) >= min_pts:
                        vecinos.extend(vecinos_del_vecino)
                
                if cluster_id[vecino] == -1:
                    cluster_id[vecino] = cluster_actual
                
                j += 1
            
            cluster_actual += 1
    
    return cluster_id

def visualizar_dbscan(puntos, cluster_ids, titulo):
    """
    Visualiza los resultados de DBSCAN
    """
    plt.figure(figsize=(10, 10))
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, punto in enumerate(puntos):
        if cluster_ids[i] == -1:
            color = 'black'  # Ruido
            marker = 'x'
        else:
            color = colores[cluster_ids[i] % len(colores)]
            marker = 'o'
        
        plt.scatter(punto[0], punto[1], c=color, s=100, marker=marker, alpha=0.7)
        plt.annotate(nombres[i], (punto[0], punto[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True, alpha=0.3)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Ejecutar K-means
print("=== ALGORITMO K-MEANS ===")
centroides_iniciales = [datos['A'], datos['D'], datos['G']]
print(f"Centroides iniciales: A{datos['A']}, D{datos['D']}, G{datos['G']}")

historial_clusters, historial_centroides = kmeans_manual(puntos, centroides_iniciales, 3)

# Ejecutar DBSCAN
print("\n=== ALGORITMO DBSCAN ===")
print("\nCon eps=2 y minPts=2:")
cluster_ids_2 = dbscan_manual(puntos, eps=2, min_pts=2)

clusters_dbscan = {}
for i, cluster_id in enumerate(cluster_ids_2):
    if cluster_id not in clusters_dbscan:
        clusters_dbscan[cluster_id] = []
    clusters_dbscan[cluster_id].append(nombres[i])

for cluster_id, puntos_cluster in clusters_dbscan.items():
    if cluster_id == -1:
        print(f"Ruido: {puntos_cluster}")
    else:
        print(f"Cluster {cluster_id + 1}: {puntos_cluster}")

visualizar_dbscan(puntos, cluster_ids_2, "DBSCAN con eps=2, minPts=2")

print("\nCon eps=10 y minPts=2:")
cluster_ids_10 = dbscan_manual(puntos, eps=10, min_pts=2)

clusters_dbscan_10 = {}
for i, cluster_id in enumerate(cluster_ids_10):
    if cluster_id not in clusters_dbscan_10:
        clusters_dbscan_10[cluster_id] = []
    clusters_dbscan_10[cluster_id].append(nombres[i])

for cluster_id, puntos_cluster in clusters_dbscan_10.items():
    if cluster_id == -1:
        print(f"Ruido: {puntos_cluster}")
    else:
        print(f"Cluster {cluster_id + 1}: {puntos_cluster}")

visualizar_dbscan(puntos, cluster_ids_10, "DBSCAN con eps=10, minPts=2")