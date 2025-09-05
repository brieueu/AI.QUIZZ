"""
Interface Gradio para o Previsor de Matriz V-Cov com RNN
Engenheiro de Software Sênior - Mercado Financeiro

Interface web moderna usando Gradio para interação com o sistema de 
previsão de matrizes de variância-covariância.

Autor: Engenheiro de Software Sênior
Especialização: Python, TensorFlow/Keras, Análise Quantitativa
"""

import gradio as gr
import plotly.graph_objects as go
from vcov_predictor import VCovPredictor


class GradioInterface:
    """Interface Gradio para o sistema de previsão V-Cov."""
    
    def __init__(self):
        """Inicialização da interface."""
        self.predictor = VCovPredictor()
    
    def predict_wrapper(self, tickers, period, window):
        """
        Wrapper da função de previsão para Gradio.
        
        Args:
            tickers (str): Tickers separados por vírgula
            period (int): Período histórico em anos  
            window (int): Janela V-Cov em dias
            
        Yields:
            tuple: (heatmap_vcov, heatmap_corr, resultado_markdown)
        """
        # Executar previsão
        result = self.predictor.predict_vcov_matrix(
            tickers, int(period), int(window)
        )
        
        if not result['success']:
            # Em caso de erro - retornar None para os gráficos e erro no texto
            return None, None, result['result_text']
        
        # Sucesso - criar visualizações
        vcov_fig = self._create_vcov_heatmap(
            result['vcov_matrix'], 
            result['tickers']
        )
        
        corr_fig = self._create_correlation_heatmap(
            result['correlation_matrix'], 
            result['tickers']
        )
        
        # Resultado final
        return vcov_fig, corr_fig, result['result_text']
    
    def _create_vcov_heatmap(self, predicted_vcov, tickers):
        """Cria heatmap da matriz de variância-covariância."""
        fig = go.Figure(data=go.Heatmap(
            z=predicted_vcov,
            x=tickers,
            y=tickers,
            colorscale='RdYlBu_r',
            showscale=True,
            text=[[f"{predicted_vcov[i,j]:.2e}" for j in range(len(tickers))] 
                  for i in range(len(tickers))],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🔥 Matriz de Variância-Covariância (Anualizada)",
            xaxis_title="Ativos",
            yaxis_title="Ativos",
            font=dict(size=12),
            height=500,
            width=600
        )
        
        return fig

    def _create_correlation_heatmap(self, corr_matrix, tickers):
        """Cria heatmap da matriz de correlação."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=tickers,
            y=tickers,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            showscale=True,
            text=[[f"{corr_matrix[i,j]:.3f}" for j in range(len(tickers))] 
                  for i in range(len(tickers))],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🎯 Matriz de Correlações",
            xaxis_title="Ativos",
            yaxis_title="Ativos",
            font=dict(size=12),
            height=500,
            width=600
        )
        
        return fig
    
    def create_interface(self):
        """Cria a interface Gradio completa."""
        
        with gr.Blocks(
            title="🚀 Previsor de Matriz V-Cov com RNN",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            """
        ) as demo:
            
            gr.Markdown("# 🚀 Previsor de Matriz V-Cov com RNN")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Configurações")
                    
                    tickers_input = gr.Textbox(
                        label="Tickers (separados por vírgula)",
                        value="PETR4.SA, VALE3.SA, ITUB4.SA",
                        placeholder="Ex: AAPL, GOOGL, MSFT",
                        info="Digite os códigos dos ativos financeiros separados por vírgula"
                    )
                    
                    period_input = gr.Textbox(
                        label="Período histórico (anos)",
                        value="5",
                        placeholder="Ex: 5",
                        info="Quantidade de anos de dados históricos para análise"
                    )
                    
                    window_input = gr.Textbox(
                        label="Janela V-Cov (dias)",
                        value="90",
                        placeholder="Ex: 90",
                        info="Janela deslizante para cálculo das matrizes de variância-covariância"
                    )
                    
                    predict_btn = gr.Button(
                        "Matriz V-Cov",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("## Resultados")
                    
                    with gr.Tabs():
                        with gr.TabItem("📋 Relatório Detalhado"):
                            result_markdown = gr.Markdown(
                                value="Aguardando previsão...",
                                height=400
                            )
                        
                        with gr.TabItem("🔥 Matriz V-Cov"):
                            vcov_plot = gr.Plot(
                                label="Heatmap da Matriz de Variância-Covariância"
                            )
                        
                        with gr.TabItem("🎯 Correlações"):
                            corr_plot = gr.Plot(
                                label="Heatmap da Matriz de Correlações"
                            )
            
            # Conectar eventos
            predict_btn.click(
                fn=self.predict_wrapper,
                inputs=[tickers_input, period_input, window_input],
                outputs=[vcov_plot, corr_plot, result_markdown],
                show_progress=True
            )
        
        return demo
    
    def launch(self, **kwargs):
        """Lança a interface Gradio."""
        demo = self.create_interface()
        return demo.launch(**kwargs)


# Função de conveniência para criar a interface
def create_gradio_interface():
    """Cria e retorna uma instância da interface Gradio."""
    interface = GradioInterface()
    return interface.create_interface()
