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
    
    def alpha_weighting_wrapper(self, tickers_input, benchmark, risk_free_rate):
        """Wrapper para análise de ponderação alfa."""
        try:
            # Converter taxa de juros para float
            risk_free_rate = float(risk_free_rate) / 100 if risk_free_rate else 0.02
            
            # Realizar análise alfa
            alpha_result = self.predictor.calculate_alpha_weighted_portfolio(
                tickers_input=tickers_input,
                benchmark=benchmark,
                risk_free_rate=risk_free_rate
            )
            
            # Formatar resultado
            alpha_text = self.predictor.format_alpha_results(alpha_result)
            
            # Criar gráfico de pesos
            weights_fig = self._create_weights_chart(alpha_result)
            
            # Criar gráfico alfa vs beta
            alpha_beta_fig = self._create_alpha_beta_scatter(alpha_result)
            
            return weights_fig, alpha_beta_fig, alpha_text
            
        except Exception as e:
            error_msg = f"❌ **Erro na análise alfa**: {str(e)}"
            empty_fig = go.Figure()
            return empty_fig, empty_fig, error_msg
    
    def chatgpt_insights_wrapper(self, tickers, openai_key, analysis_type):
        """Wrapper para insights do ChatGPT."""
        try:
            if not openai_key.strip():
                return "❌ **Erro**: Por favor, insira sua API key do OpenAI"
            
            # Configurar API key
            success = self.predictor.configure_chatgpt(openai_key.strip())
            if not success:
                return "❌ **Erro**: API key inválida ou erro de conexão"
            
            tickers_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
            
            if analysis_type == "Comentário de Mercado":
                context = f"Análise de mercado para portfólio de {len(tickers_list)} ativos"
                return self.predictor.get_market_commentary(tickers_list, context)
            else:
                return "❌ **Erro**: Tipo de análise não suportado"
                
        except Exception as e:
            return f"❌ **Erro nos insights**: {str(e)}"
    
    def vcov_insights_wrapper(self, tickers, period, window, openai_key):
        """Wrapper para insights V-Cov do ChatGPT."""
        try:
            if not openai_key.strip():
                return "❌ **Erro**: Por favor, insira sua API key do OpenAI para insights"
            
            # Configurar API key
            success = self.predictor.configure_chatgpt(openai_key.strip())
            if not success:
                return "❌ **Erro**: API key inválida ou erro de conexão"
            
            # Executar previsão V-Cov primeiro
            result = self.predictor.predict_vcov_matrix(
                tickers, int(period), int(window)
            )
            
            if not result['success']:
                return f"❌ **Erro na previsão**: {result['result_text']}"
            
            # Gerar insights
            additional_data = {
                'period_years': int(period),
                'window_days': int(window),
                'prediction_date': result.get('prediction_date', 'N/A')
            }
            
            return self.predictor.get_vcov_insights(
                result['vcov_matrix'], 
                result['tickers'],
                additional_data
            )
            
        except Exception as e:
            return f"❌ **Erro nos insights V-Cov**: {str(e)}"
    
    def alpha_insights_wrapper(self, tickers_input, benchmark, risk_free_rate, openai_key):
        """Wrapper para insights de ponderação alfa do ChatGPT."""
        try:
            if not openai_key.strip():
                return "❌ **Erro**: Por favor, insira sua API key do OpenAI para insights"
            
            # Configurar API key
            success = self.predictor.configure_chatgpt(openai_key.strip())
            if not success:
                return "❌ **Erro**: API key inválida ou erro de conexão"
            
            # Converter taxa de juros
            risk_free_rate = float(risk_free_rate) / 100 if risk_free_rate else 0.02
            
            # Executar análise alfa
            alpha_result = self.predictor.calculate_alpha_weighted_portfolio(
                tickers_input=tickers_input,
                benchmark=benchmark,
                risk_free_rate=risk_free_rate
            )
            
            # Gerar insights
            return self.predictor.get_alpha_insights(alpha_result)
            
        except Exception as e:
            return f"❌ **Erro nos insights alfa**: {str(e)}"
    
    def _create_weights_chart(self, alpha_result):
        """Cria gráfico de barras com os pesos calculados."""
        if 'error' in alpha_result or not alpha_result['weights']:
            return go.Figure()
        
        tickers = list(alpha_result['weights'].keys())
        weights = list(alpha_result['weights'].values())
        alphas = [alpha_result['alphas'].get(t, 0) for t in tickers]
        
        # Cores baseadas no alfa
        colors = ['green' if alpha > 0 else 'red' for alpha in alphas]
        
        fig = go.Figure(data=go.Bar(
            x=tickers,
            y=[w * 100 for w in weights],  # Converter para porcentagem
            marker_color=colors,
            text=[f"{w:.1%}" for w in weights],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="🎯 Ponderação Alfa - Pesos do Portfólio",
            xaxis_title="Ativos",
            yaxis_title="Peso (%)",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_alpha_beta_scatter(self, alpha_result):
        """Cria gráfico de dispersão alfa vs beta."""
        if 'error' in alpha_result or not alpha_result['alphas']:
            return go.Figure()
        
        tickers = list(alpha_result['alphas'].keys())
        alphas = list(alpha_result['alphas'].values())
        betas = [alpha_result['betas'].get(t, 0) for t in tickers]
        weights = [alpha_result['weights'].get(t, 0) for t in tickers]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=betas,
            y=alphas,
            mode='markers+text',
            marker=dict(
                size=[w * 1000 for w in weights],  # Tamanho proporcional ao peso
                color=alphas,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Alfa")
            ),
            text=tickers,
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>" +
                         "Beta: %{x:.3f}<br>" +
                         "Alfa: %{y:.3f}<br>" +
                         "<extra></extra>"
        ))
        
        # Linhas de referência
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="📊 Análise Alfa vs Beta",
            xaxis_title="Beta (Sensibilidade ao Mercado)",
            yaxis_title="Alfa (Retorno em Excesso)",
            height=500,
            annotations=[
                dict(x=0.02, y=0.98, xref="paper", yref="paper",
                     text="Tamanho da bolha = Peso no portfólio", 
                     showarrow=False, font=dict(size=10))
            ]
        )
        
        return fig
    
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
            title="Previsor de Matriz V-Cov com RNN",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
            }
            """
        ) as demo:
            
            gr.Markdown("# 🚀 Previsor de Matriz V-Cov com RNN + Ponderação Alfa")
            
            with gr.Tabs():
                with gr.TabItem("📈 Predição V-Cov"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Configurações V-Cov")
                            
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
                                "🔮 Gerar Predição V-Cov",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("## Resultados V-Cov")
                            
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
                
                with gr.TabItem("🎯 Ponderação Alfa"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Configurações Ponderação Alfa")
                            
                            alpha_tickers_input = gr.Textbox(
                                label="Tickers para análise alfa",
                                value="AAPL, GOOGL, MSFT, AMZN, TSLA",
                                placeholder="Ex: AAPL, GOOGL, MSFT",
                                info="Ativos para calcular ponderação baseada em alfa"
                            )
                            
                            benchmark_input = gr.Textbox(
                                label="Benchmark",
                                value="^GSPC",
                                placeholder="^GSPC (S&P 500)",
                                info="Índice de referência para cálculo do alfa"
                            )
                            
                            risk_free_input = gr.Textbox(
                                label="Taxa livre de risco (%)",
                                value="2.0",
                                placeholder="2.0",
                                info="Taxa livre de risco anual em porcentagem"
                            )
                            
                            alpha_btn = gr.Button(
                                "🎯 Calcular Ponderação Alfa",
                                variant="secondary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("## Resultados Ponderação Alfa")
                            
                            with gr.Tabs():
                                with gr.TabItem("📋 Análise Detalhada"):
                                    alpha_result_markdown = gr.Markdown(
                                        value="Aguardando análise alfa...",
                                        height=400
                                    )
                                
                                with gr.TabItem("⚖️ Pesos do Portfólio"):
                                    weights_plot = gr.Plot(
                                        label="Ponderação dos Ativos"
                                    )
                                
                                with gr.TabItem("📊 Alfa vs Beta"):
                                    alpha_beta_plot = gr.Plot(
                                        label="Dispersão Alfa vs Beta"
                                    )
                
                with gr.TabItem("🤖 Insights ChatGPT"):
                    gr.Markdown("## 🧠 Análises Inteligentes com ChatGPT")
                    gr.Markdown("*Configure sua API key do OpenAI para obter insights profissionais sobre seus dados financeiros*")
                    
                    with gr.Row():
                        openai_key_input = gr.Textbox(
                            label="🔑 API Key OpenAI",
                            placeholder="sk-...",
                            type="password",
                            info="Sua chave da API OpenAI para gerar insights"
                        )
                    
                    with gr.Tabs():
                        with gr.TabItem("💬 Comentário de Mercado"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    market_tickers_input = gr.Textbox(
                                        label="Ativos para análise",
                                        value="AAPL, GOOGL, MSFT, AMZN",
                                        placeholder="Ex: AAPL, MSFT, TSLA"
                                    )
                                    
                                    market_analysis_type = gr.Dropdown(
                                        label="Tipo de análise",
                                        choices=["Comentário de Mercado"],
                                        value="Comentário de Mercado"
                                    )
                                    
                                    market_btn = gr.Button(
                                        "📰 Gerar Comentário",
                                        variant="secondary",
                                        size="lg"
                                    )
                                
                                with gr.Column(scale=2):
                                    market_insights_output = gr.Markdown(
                                        value="Configure a API key e clique em 'Gerar Comentário'...",
                                        height=500
                                    )
                        
                        with gr.TabItem("🔮 Insights V-Cov"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    vcov_insights_tickers = gr.Textbox(
                                        label="Tickers para V-Cov",
                                        value="AAPL, GOOGL, MSFT",
                                        placeholder="Ex: AAPL, GOOGL, MSFT"
                                    )
                                    
                                    vcov_insights_period = gr.Textbox(
                                        label="Período (anos)",
                                        value="3",
                                        placeholder="Ex: 3"
                                    )
                                    
                                    vcov_insights_window = gr.Textbox(
                                        label="Janela (dias)",
                                        value="60",
                                        placeholder="Ex: 60"
                                    )
                                    
                                    vcov_insights_btn = gr.Button(
                                        "🔍 Analisar V-Cov",
                                        variant="secondary",
                                        size="lg"
                                    )
                                
                                with gr.Column(scale=2):
                                    vcov_insights_output = gr.Markdown(
                                        value="Configure a API key e parâmetros, depois clique em 'Analisar V-Cov'...",
                                        height=500
                                    )
                        
                        with gr.TabItem("🎯 Insights Alfa"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    alpha_insights_tickers = gr.Textbox(
                                        label="Tickers para análise alfa",
                                        value="AAPL, GOOGL, MSFT, AMZN, TSLA",
                                        placeholder="Ex: AAPL, GOOGL, MSFT"
                                    )
                                    
                                    alpha_insights_benchmark = gr.Textbox(
                                        label="Benchmark",
                                        value="^GSPC",
                                        placeholder="^GSPC"
                                    )
                                    
                                    alpha_insights_risk_free = gr.Textbox(
                                        label="Taxa livre de risco (%)",
                                        value="2.0",
                                        placeholder="2.0"
                                    )
                                    
                                    alpha_insights_btn = gr.Button(
                                        "🧮 Analisar Alfa",
                                        variant="secondary",
                                        size="lg"
                                    )
                                
                                with gr.Column(scale=2):
                                    alpha_insights_output = gr.Markdown(
                                        value="Configure a API key e parâmetros, depois clique em 'Analisar Alfa'...",
                                        height=500
                                    )
            
            # Conectar eventos V-Cov
            predict_btn.click(
                fn=self.predict_wrapper,
                inputs=[tickers_input, period_input, window_input],
                outputs=[vcov_plot, corr_plot, result_markdown],
                show_progress=True
            )
            
            # Conectar eventos Ponderação Alfa
            alpha_btn.click(
                fn=self.alpha_weighting_wrapper,
                inputs=[alpha_tickers_input, benchmark_input, risk_free_input],
                outputs=[weights_plot, alpha_beta_plot, alpha_result_markdown],
                show_progress=True
            )
            
            # Conectar eventos ChatGPT
            market_btn.click(
                fn=self.chatgpt_insights_wrapper,
                inputs=[market_tickers_input, openai_key_input, market_analysis_type],
                outputs=[market_insights_output],
                show_progress=True
            )
            
            vcov_insights_btn.click(
                fn=self.vcov_insights_wrapper,
                inputs=[vcov_insights_tickers, vcov_insights_period, vcov_insights_window, openai_key_input],
                outputs=[vcov_insights_output],
                show_progress=True
            )
            
            alpha_insights_btn.click(
                fn=self.alpha_insights_wrapper,
                inputs=[alpha_insights_tickers, alpha_insights_benchmark, alpha_insights_risk_free, openai_key_input],
                outputs=[alpha_insights_output],
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
