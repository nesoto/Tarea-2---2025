import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_learning_curves():
    """
    Grafica las curvas de aprendizaje para Q-learning y SARSA
    """
    try:
        # Cargar datos de los archivos de rewards
        data_qlearning_env1 = pd.read_csv('Rewards_Qlearning_Env1.txt', names=['Episode', 'Reward'])
        data_qlearning_env2 = pd.read_csv('Rewards_Qlearning_Env2.txt', names=['Episode', 'Reward'])
        data_sarsa_env1 = pd.read_csv('Rewards_SARSA_Env1.txt', names=['Episode', 'Reward'])
        data_sarsa_env2 = pd.read_csv('Rewards_SARSA_Env2.txt', names=['Episode', 'Reward'])
        
        # Crear subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Q-learning Ambiente 1
        ax1.plot(data_qlearning_env1['Episode'], data_qlearning_env1['Reward'], 'b-', alpha=0.7)
        ax1.set_title('Q-learning - Ambiente 1 (Grid 3x4)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Reward Acumulado')
        ax1.grid(True, alpha=0.3)
        
        # Q-learning Ambiente 2
        ax2.plot(data_qlearning_env2['Episode'], data_qlearning_env2['Reward'], 'b-', alpha=0.7)
        ax2.set_title('Q-learning - Ambiente 2 (Cliff Walking)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Reward Acumulado')
        ax2.grid(True, alpha=0.3)
        
        # SARSA Ambiente 1
        ax3.plot(data_sarsa_env1['Episode'], data_sarsa_env1['Reward'], 'r-', alpha=0.7)
        ax3.set_title('SARSA - Ambiente 1 (Grid 3x4)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Episodio')
        ax3.set_ylabel('Reward Acumulado')
        ax3.grid(True, alpha=0.3)
        
        # SARSA Ambiente 2
        ax4.plot(data_sarsa_env2['Episode'], data_sarsa_env2['Reward'], 'r-', alpha=0.7)
        ax4.set_title('SARSA - Ambiente 2 (Cliff Walking)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Reward Acumulado')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfico comparativo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Ambiente 1 - Comparación
        window = 100  # Ventana para promedio móvil
        q_smooth_1 = data_qlearning_env1['Reward'].rolling(window=window, min_periods=1).mean()
        s_smooth_1 = data_sarsa_env1['Reward'].rolling(window=window, min_periods=1).mean()
        
        ax1.plot(data_qlearning_env1['Episode'], q_smooth_1, 'b-', label='Q-learning', linewidth=2)
        ax1.plot(data_sarsa_env1['Episode'], s_smooth_1, 'r-', label='SARSA', linewidth=2)
        ax1.set_title('Comparación Ambiente 1 (Promedio móvil)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Reward Acumulado')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ambiente 2 - Comparación
        q_smooth_2 = data_qlearning_env2['Reward'].rolling(window=window, min_periods=1).mean()
        s_smooth_2 = data_sarsa_env2['Reward'].rolling(window=window, min_periods=1).mean()
        
        ax2.plot(data_qlearning_env2['Episode'], q_smooth_2, 'b-', label='Q-learning', linewidth=2)
        ax2.plot(data_sarsa_env2['Episode'], s_smooth_2, 'r-', label='SARSA', linewidth=2)
        ax2.set_title('Comparación Ambiente 2 (Promedio móvil)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Reward Acumulado')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análisis estadístico
        print("=== ANÁLISIS ESTADÍSTICO ===")
        print("\nAmbiente 1 (Grid 3x4):")
        print(f"Q-learning - Reward promedio últimos 500 episodios: {data_qlearning_env1['Reward'].tail(500).mean():.2f}")
        print(f"SARSA - Reward promedio últimos 500 episodios: {data_sarsa_env1['Reward'].tail(500).mean():.2f}")
        
        print("\nAmbiente 2 (Cliff Walking):")
        print(f"Q-learning - Reward promedio últimos 500 episodios: {data_qlearning_env2['Reward'].tail(500).mean():.2f}")
        print(f"SARSA - Reward promedio últimos 500 episodios: {data_sarsa_env2['Reward'].tail(500).mean():.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: No se pudieron cargar los archivos de datos. {e}")
        print("Asegúrate de haber ejecutado el programa C++ para ambos algoritmos y ambientes.")

def run_cpp_experiments():
    """
    Instrucciones para ejecutar los experimentos
    """
    print("=== INSTRUCCIONES PARA EJECUTAR EXPERIMENTOS ===")
    print("1. Compila el programa C++:")
    print("   g++ -o tutorial tutorial.cpp")
    print("\n2. Ejecuta los experimentos modificando las siguientes variables en tutorial.cpp:")
    print("   - Para Q-learning Ambiente 1: algorithm=1, environment=1")
    print("   - Para Q-learning Ambiente 2: algorithm=1, environment=2") 
    print("   - Para SARSA Ambiente 1: algorithm=2, environment=1")
    print("   - Para SARSA Ambiente 2: algorithm=2, environment=2")
    print("\n3. Renombra el archivo Rewards.txt después de cada ejecución:")
    print("   - Rewards_Qlearning_Env1.txt")
    print("   - Rewards_Qlearning_Env2.txt")
    print("   - Rewards_SARSA_Env1.txt")
    print("   - Rewards_SARSA_Env2.txt")
    print("\n4. Ejecuta este script para generar las visualizaciones.")

if __name__ == "__main__":
    # Ejecutar visualizaciones si existen los archivos
    plot_learning_curves()
    
    # Mostrar instrucciones
    run_cpp_experiments()