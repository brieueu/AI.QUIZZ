"""
Integração com Google AI Studio (Gemini) para Insights Financeiros
Engenheiro de Software Sênior - Mercado Financeiro

Sistema de análise inteligente usando Gemini para gerar insights
sobre dados financeiros, matrizes V-Cov e performance de portfólios.

Autor: Engenheiro de Software Sênior
Especialização: Python, IA, Análise Quantitativa
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import google.generativeai as genai


class GeminiInsights:
    """
    Sistema de insights financeiros usando Google AI Studio (Gemini).
    
    Gera análises inteligentes sobre:
    - Matrizes de variância-covariância
    - Performance de portfólios
    - Comentários de mercado
    - Avaliações de risco
    """
    
    def __init__(self):
        """Inicializa o cliente Gemini AI."""
        self.client = None
        self.model = None
        self.api_key = None
        
        # Tentar carregar API key do ambiente
        api_key = os.getenv('GOOGLE_AI_API_KEY')
        if api_key:
            if self.set_api_key(api_key):
                print("✅ API Key do Google AI Studio carregada do ambiente!")
            else:
                print("⚠️ Erro ao configurar API Key do ambiente")
        else:
            print("⚠️ API Key do Google AI Studio não encontrada. Configure GOOGLE_AI_API_KEY ou forneça a chave.")
    
    def _format_financial_data(self, data: Dict) -> str:
        """
        Formata dados financeiros para envio ao Gemini.
        
        Args:
            data: Dicionário com dados financeiros
            
        Returns:
            String formatada para análise
        """
        formatted_text = "DADOS FINANCEIROS PARA ANÁLISE:\n"
        formatted_text += "=" * 50 + "\n\n"
        
        # Adicionar timestamp
        formatted_text += f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Adicionar dados formatados
        for key, value in data.items():
            if isinstance(value, dict):
                formatted_text += f"{key.upper()}:\n"
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        formatted_text += f"  - {sub_key}: {sub_value:.4f}\n"
                    else:
                        formatted_text += f"  - {sub_key}: {sub_value}\n"
                formatted_text += "\n"
            elif isinstance(value, (list, tuple)):
                formatted_text += f"{key.upper()}: {', '.join(map(str, value))}\n\n"
            else:
                formatted_text += f"{key.upper()}: {value}\n\n"
        
        return formatted_text
    
    def analyze_vcov_matrix(self, vcov_matrix: np.ndarray, tickers: List[str], 
                           additional_data: Dict = None) -> str:
        """
        Analisa matriz de variância-covariância usando Gemini.
        
        Args:
            vcov_matrix: Matriz V-Cov
            tickers: Lista de tickers
            additional_data: Dados adicionais para contexto
            
        Returns:
            Análise textual do Gemini
        """
        if not self.model:
            return "❌ Google AI Studio não disponível - configure a API key"
        
        try:
            # Preparar dados para análise
            data = {
                "tickers": tickers,
                "matriz_vcov": {
                    f"{tickers[i]}_{tickers[j]}": vcov_matrix[i, j]
                    for i in range(len(tickers))
                    for j in range(len(tickers))
                },
                "volatilidades": {
                    ticker: np.sqrt(vcov_matrix[i, i])
                    for i, ticker in enumerate(tickers)
                },
                "correlacoes_maximas": {
                    "par": f"{tickers[np.unravel_index(np.argmax(np.abs(np.corrcoef(vcov_matrix) - np.eye(len(tickers)))), (len(tickers), len(tickers)))[0]]}_{tickers[np.unravel_index(np.argmax(np.abs(np.corrcoef(vcov_matrix) - np.eye(len(tickers)))), (len(tickers), len(tickers)))[1]]}",
                    "valor": np.max(np.abs(np.corrcoef(vcov_matrix) - np.eye(len(tickers))))
                }
            }
            
            if additional_data:
                data.update(additional_data)
            
            formatted_data = self._format_financial_data(data)
            
            # Prompt especializado para análise V-Cov
            prompt = f"""
            {formatted_data}
            
            Como especialista em análise quantitativa e gestão de risco, forneça insights detalhados sobre esta matriz de variância-covariância:
            
            1. **ANÁLISE DE RISCO**: Identifique os ativos mais e menos voláteis
            2. **CORRELAÇÕES**: Analise as relações entre os ativos e suas implicações
            3. **DIVERSIFICAÇÃO**: Avalie o potencial de diversificação do portfólio
            4. **CONCENTRAÇÃO DE RISCO**: Identifique possíveis concentrações perigosas
            5. **RECOMENDAÇÕES**: Sugira ajustes para otimização do portfólio
            
            Seja específico, técnico mas acessível. Use emojis para melhor visualização.
            """
            
            response = self.model.generate_content(prompt)
            return f"## 🔮 **Insights V-Cov - Gemini AI**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ **Erro ao gerar insights V-Cov**: {str(e)}"
    
    def analyze_alpha_portfolio(self, alpha_result: Dict) -> str:
        """
        Analisa resultado de ponderação alfa usando Gemini.
        
        Args:
            alpha_result: Resultado do cálculo de alfa
            
        Returns:
            Análise textual do Gemini
        """
        if not self.model:
            return "❌ Google AI Studio não disponível - configure a API key"
        
        try:
            formatted_data = self._format_financial_data(alpha_result)
            
            # Prompt especializado para análise alfa
            prompt = f"""
            {formatted_data}
            
            Como especialista em análise de performance de portfólios, analise estes resultados de alfa/beta:
            
            1. **ANÁLISE DE ALFA**: Interprete os valores de alfa e seu significado
            2. **EXPOSIÇÃO BETA**: Avalie os níveis de exposição ao mercado
            3. **PERFORMANCE**: Compare a performance relativa entre ativos
            4. **SELEÇÃO DE ATIVOS**: Identifique os melhores geradores de alfa
            5. **ESTRATÉGIA**: Recomende estratégias baseadas nos alfas calculados
            
            Seja específico sobre implicações práticas para investimento. Use emojis.
            """
            
            response = self.model.generate_content(prompt)
            return f"## 🎯 **Insights Alpha - Gemini AI**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ **Erro ao gerar insights alfa**: {str(e)}"
    
    def generate_market_commentary(self, tickers: List[str], context: str = "") -> str:
        """
        Gera comentário de mercado usando Gemini.
        
        Args:
            tickers: Lista de tickers para análise
            context: Contexto adicional
            
        Returns:
            Comentário de mercado
        """
        if not self.model:
            return "❌ Google AI Studio não disponível - configure a API key"
        
        try:
            # Preparar dados
            data = {
                "ativos_analisados": tickers,
                "contexto": context or "Análise geral de mercado",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            formatted_data = self._format_financial_data(data)
            
            prompt = f"""
            {formatted_data}
            
            Como analista de mercado sênior, gere um comentário profissional sobre estes ativos:
            
            1. **CENÁRIO ATUAL**: Visão geral do mercado para estes ativos
            2. **TENDÊNCIAS**: Identifique tendências relevantes
            3. **RISCOS E OPORTUNIDADES**: Analise fatores de risco e oportunidades
            4. **SETORES**: Comente sobre os setores representados
            5. **OUTLOOK**: Perspectivas de curto e médio prazo
            
            Mantenha tom profissional mas acessível. Use emojis para organização.
            """
            
            response = self.model.generate_content(prompt)
            return f"## 📰 **Comentário de Mercado - Gemini AI**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ **Erro ao gerar comentário**: {str(e)}"
    
    def generate_risk_assessment(self, tickers: List[str], portfolio_data: Dict) -> str:
        """
        Gera avaliação de risco usando Gemini.
        
        Args:
            tickers: Lista de tickers
            portfolio_data: Dados do portfólio
            
        Returns:
            Avaliação de risco
        """
        if not self.model:
            return "❌ Google AI Studio não disponível - configure a API key"
        
        try:
            data = {
                "ativos": tickers,
                "dados_portfolio": portfolio_data
            }
            
            formatted_data = self._format_financial_data(data)
            
            prompt = f"""
            {formatted_data}
            
            Como especialista em gestão de risco, forneça uma avaliação abrangente:
            
            1. **PERFIL DE RISCO**: Classificação geral do portfólio
            2. **MÉTRICAS**: Análise das principais métricas de risco
            3. **CONCENTRAÇÃO**: Avaliação de concentrações setoriais/geográficas
            4. **CENÁRIOS**: Análise de cenários adversos
            5. **MITIGAÇÃO**: Estratégias para redução de risco
            
            Seja específico em recomendações práticas. Use emojis.
            """
            
            response = self.model.generate_content(prompt)
            return f"## ⚠️ **Avaliação de Risco - Gemini AI**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ **Erro ao gerar avaliação**: {str(e)}"
    
    def set_api_key(self, api_key: str) -> bool:
        """
        Configura a API key do Google AI Studio.
        
        Args:
            api_key: API key do Google AI Studio
            
        Returns:
            bool: True se configuração bem-sucedida
        """
        # Lista de modelos para tentar (do mais recente para o mais antigo)
        models_to_try = [
            'gemini-1.5-pro',      # Melhor para análises complexas
            'gemini-1.5-flash',    # Mais rápido
            'gemini-2.0-flash-exp' # Experimental mais recente
        ]
        
        try:
            genai.configure(api_key=api_key)
            
            # Tentar configurar um modelo disponível
            for model_name in models_to_try:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.api_key = api_key
                    print(f"✅ Nova API key configurada com modelo {model_name}!")
                    return True
                except Exception as model_error:
                    print(f"⚠️ Modelo {model_name} não disponível: {str(model_error)}")
                    continue
            
            # Se nenhum modelo específico funcionou, tentar modelo genérico
            print("⚠️ Tentando modelo padrão...")
            self.model = genai.GenerativeModel('gemini-pro')  # fallback
            self.api_key = api_key
            print("✅ Nova API key do Google AI Studio configurada com modelo padrão!")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao configurar API key: {str(e)}")
            return False
    
    def list_available_models(self) -> str:
        """
        Lista todos os modelos disponíveis do Google AI Studio.
        
        Returns:
            str: Lista formatada dos modelos disponíveis
        """
        if not self.api_key:
            return "❌ API Key não configurada. Configure primeiro para listar modelos."
        
        try:
            models = genai.list_models()
            
            result = "## 🤖 **Modelos Gemini Disponíveis**\n\n"
            result += "| Modelo | Capacidades | Limites |\n"
            result += "|--------|-------------|----------|\n"
            
            for model in models:
                model_name = model.name.replace('models/', '')
                
                # Informações específicas por modelo (atualizadas para novos modelos)
                if 'gemini-1.5-pro' in model_name:
                    capabilities = "Texto, Análise Avançada, 2M tokens"
                    limits = "2 req/min (free)"
                elif 'gemini-1.5-flash' in model_name:
                    capabilities = "Texto, Análise Rápida, 1M tokens"
                    limits = "15 req/min (free)"
                elif 'gemini-2.0-flash' in model_name:
                    capabilities = "Texto, Multimodal, Mais Recente"
                    limits = "10 req/min (free)"
                elif 'vision' in model_name or 'pro-vision' in model_name:
                    capabilities = "Texto, Imagem, Análise Visual"
                    limits = "60 req/hour"
                else:
                    capabilities = "Texto, Multimodal"
                    limits = "Varia"
                
                result += f"| {model_name} | {capabilities} | {limits} |\n"
            
            result += "\n### 💡 **Recomendações:**\n"
            result += "- **gemini-1.5-pro**: Melhor para análises financeiras complexas (2M tokens)\n"
            result += "- **gemini-1.5-flash**: Análises rápidas e eficientes (1M tokens)\n"
            result += "- **gemini-2.0-flash**: Modelo mais recente com capacidades expandidas\n"
            result += "- **Limites**: Respeite os rate limits para evitar erros (especialmente no plano gratuito)\n"
            result += "- **Tokens**: Modelos 1.5 suportam contextos muito maiores que versões anteriores\n"
            
            return result
            
        except Exception as e:
            return f"❌ Erro ao listar modelos: {str(e)}\n\nVerifique:\n- Conectividade com internet\n- Validade da API key\n- Status dos serviços Google AI"


# Função de conveniência para criar instância
def create_gemini_insights(api_key: Optional[str] = None) -> GeminiInsights:
    """
    Cria uma instância do GeminiInsights.
    
    Args:
        api_key: API key opcional do Google AI Studio
        
    Returns:
        GeminiInsights: Instância configurada
    """
    insights = GeminiInsights()
    if api_key:
        insights.set_api_key(api_key)
    return insights


# Exemplo de uso
def example_usage():
    """Exemplo de como usar o sistema de insights."""
    # Criar instância
    insights = create_gemini_insights("sua_api_key_aqui")
    
    # Exemplo de análise V-Cov
    import numpy as np
    vcov_matrix = np.random.rand(3, 3)
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    analysis = insights.analyze_vcov_matrix(vcov_matrix, tickers)
    print(analysis)
