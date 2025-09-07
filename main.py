"""
Previsor de Matriz V-Cov com RNN - Ponto de Entrada Principal
Sistema avançado para previsão de matrizes de variância-covariância
usando decomposição de Cholesky e redes neurais LSTM.
"""

from gradio_interface import GradioInterface


def main():
    """Função principal para lançar a aplicação."""
    
    print("🚀 Iniciando Previsor de Matriz V-Cov com RNN...")
    print("📊 Carregando interface Gradio...")
    
    # Criar e lançar a interface
    interface = GradioInterface()
    
    # Configurações de lançamento
    launch_config = {
        "server_name": "0.0.0.0",    # Acessível externamente
        "server_port": 7862,         # Porta alternativa
        "share": False,              # True para link público via gradio.live
        "debug": True,               # Modo debug para desenvolvimento
        "show_error": True,          # Mostrar erros detalhados
        "quiet": False,              # Logs detalhados
        "favicon_path": None,        # Ícone personalizado (opcional)
        "app_kwargs": {              # Configurações adicionais do FastAPI
            "docs_url": None,        # Desabilitar documentação automática
            "redoc_url": None        # Desabilitar ReDoc
        }
    }
    
    print("🌐 Lançando aplicação web...")
    print(f"📍 Acesse: http://localhost:{launch_config['server_port']}")
    print("⚡ Pressione Ctrl+C para parar o servidor")
    
    # Lançar a interface
    interface.launch(**launch_config)


if __name__ == "__main__":
    main()
