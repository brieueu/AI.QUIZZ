"""
Módulo de Insights com Google AI Studio (Gemini)
===============================================

Este módulo integra a API do Google AI Studio (Gemini) para fornecer análises inteligentes
e insights sobre dados financeiros, matrizes V-Cov e ponderação alfa.

Author: Gabriel
Date: 2025-09-10
"""

import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class GeminiInsights:
    """
    Classe para gerar insights financeiros usando Google AI Studio (Gemini).
    
    Fornece análises contextuais sobre:
    - Matrizes de variância-covariância
    - Ponderação alfa
    - Performance de portfólios
    - Recomendações estratégicas
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa o módulo de insights.
        
        Args:
            api_key: Chave da API Google AI Studio (se não fornecida, tentará ler da env var)
        """
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            print("⚠️ API Key do Google AI Studio não encontrada. Configure GOOGLE_AI_API_KEY ou forneça a chave.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                print("✅ Conexão com Google AI Studio (Gemini) estabelecida!")
            except Exception as e:
                print(f"❌ Erro ao conectar com Google AI Studio: {e}")
                self.model = None
    
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
            analysis_data = {
                'tickers': tickers,
                'num_assets': len(tickers),
                'volatilities': {
                    ticker: np.sqrt(vcov_matrix[i, i]) * 100 
                    for i, ticker in enumerate(tickers)
                },
                'max_volatility': max([np.sqrt(vcov_matrix[i, i]) * 100 for i in range(len(tickers))]),
                'min_volatility': min([np.sqrt(vcov_matrix[i, i]) * 100 for i in range(len(tickers))]),
                'correlations': {}
            }
            
            # Calcular correlações principais
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    std_i = np.sqrt(vcov_matrix[i, i])
                    std_j = np.sqrt(vcov_matrix[j, j])
                    corr = vcov_matrix[i, j] / (std_i * std_j)
                    analysis_data['correlations'][f"{tickers[i]}-{tickers[j]}"] = corr
            
            if additional_data:
                analysis_data.update(additional_data)
            
            # Formatar dados
            formatted_data = self._format_financial_data(analysis_data)
            
            # Prompt para análise V-Cov
            prompt = f"""
Como especialista em análise quantitativa de mercados financeiros com mais de 15 anos de experiência, analise os seguintes dados de uma matriz de variância-covariância:

{formatted_data}

Forneça uma análise detalhada e prática incluindo:

1. **ANÁLISE DE RISCO:**
   - Interpretação das volatilidades dos ativos
   - Identificação dos ativos mais e menos arriscados
   - Avaliação do nível geral de risco do portfólio

2. **ANÁLISE DE CORRELAÇÃO:**
   - Padrões de correlação entre ativos
   - Identificação de correlações altas/baixas significativas
   - Implicações para diversificação de portfólio

3. **INSIGHTS ESTRATÉGICOS:**
   - Recomendações para construção de portfólio otimizado
   - Estratégias de hedge baseadas nas correlações
   - Considerações sobre timing de mercado

4. **ALERTAS E RISCOS:**
   - Potenciais armadilhas ou concentrações de risco
   - Cenários de stress que poderiam afetar o portfólio
   - Recomendações de monitoramento e controle de risco

5. **AÇÕES PRÁTICAS:**
   - Próximos passos específicos para implementação
   - Métricas-chave para acompanhamento
   - Sinais de alerta para revisão da estratégia

Seja específico, prático e foque em ações concretas que um gestor de portfólio poderia implementar imediatamente.
"""
            
            # Chamar API
            response = self.model.generate_content(prompt)
            
            return f"🤖 **ANÁLISE GEMINI - MATRIZ V-COV**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ Erro ao gerar insights: {str(e)}"
    
    def analyze_alpha_portfolio(self, alpha_result: Dict) -> str:
        """
        Analisa resultados de ponderação alfa usando Gemini.
        
        Args:
            alpha_result: Resultado da análise alfa
            
        Returns:
            Análise textual do Gemini
        """
        if not self.model:
            return "❌ Google AI Studio não disponível - configure a API key"
        
        if 'error' in alpha_result:
            return f"❌ Não é possível analisar devido a erro: {alpha_result['error']}"
        
        try:
            # Preparar dados para análise
            analysis_data = {
                'portfolio_stats': alpha_result.get('stats', {}),
                'alpha_values': alpha_result.get('alphas', {}),
                'beta_values': alpha_result.get('betas', {}),
                'weights': alpha_result.get('weights', {}),
                'tickers': alpha_result.get('tickers', [])
            }
            
            # Calcular métricas adicionais
            alphas = list(alpha_result.get('alphas', {}).values())
            betas = list(alpha_result.get('betas', {}).values())
            weights = list(alpha_result.get('weights', {}).values())
            
            analysis_data['metrics'] = {
                'positive_alpha_count': sum(1 for a in alphas if a > 0),
                'negative_alpha_count': sum(1 for a in alphas if a < 0),
                'avg_alpha': np.mean(alphas) if alphas else 0,
                'avg_beta': np.mean(betas) if betas else 0,
                'max_weight': max(weights) if weights else 0,
                'portfolio_concentration': max(weights) if weights else 0
            }
            
            formatted_data = self._format_financial_data(analysis_data)
            
            # Prompt para análise alfa
            prompt = f"""
Como especialista em gestão quantitativa de portfólios e estratégias alpha-driven, analise os seguintes resultados de ponderação baseada em alfa:

{formatted_data}

Forneça uma análise abrangente e acionável incluindo:

1. **ANÁLISE DE PERFORMANCE:**
   - Interpretação detalhada dos valores de alfa e beta
   - Identificação de ativos outperformers e underperformers
   - Avaliação da qualidade geral do portfólio

2. **ANÁLISE DE PONDERAÇÃO:**
   - Adequação dos pesos calculados
   - Análise de concentração vs diversificação
   - Riscos e benefícios da estratégia de ponderação

3. **ESTRATÉGIAS RECOMENDADAS:**
   - Como implementar esta ponderação na prática
   - Estratégias long/short baseadas nos alfas
   - Frequência ideal de rebalanceamento do portfólio

4. **CONSIDERAÇÕES DE RISCO:**
   - Limitações e pressupostos da análise de alfa
   - Riscos específicos dos ativos identificados
   - Estratégias de hedging e proteção de portfólio

5. **IMPLEMENTAÇÃO PRÁTICA:**
   - Próximos passos concretos para implementação
   - Métricas essenciais para monitoramento contínuo
   - Sinais e gatilhos para revisão da estratégia

6. **CENÁRIOS E STRESS TESTING:**
   - Como o portfólio reagiria em diferentes cenários de mercado
   - Recomendações para períodos de alta volatilidade

Foque em recomendações práticas e implementáveis para gestores de portfólio institucionais.
"""
            
            response = self.model.generate_content(prompt)
            
            return f"🤖 **ANÁLISE GEMINI - PONDERAÇÃO ALFA**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ Erro ao gerar insights alfa: {str(e)}"
    
    def generate_market_commentary(self, tickers: List[str], context: str = "") -> str:
        """
        Gera comentário de mercado contextual para os ativos.
        
        Args:
            tickers: Lista de tickers
            context: Contexto adicional
            
        Returns:
            Comentário de mercado
        """
        if not self.model:
            return "❌ Google AI Studio não disponível - configure a API key"
        
        try:
            prompt = f"""
Como analista sênior de mercados financeiros com expertise em análise fundamentalista e técnica, forneça um comentário atual e abrangente sobre os seguintes ativos: {', '.join(tickers)}

{f"Contexto adicional: {context}" if context else ""}

Estruture sua análise incluindo:

1. **VISÃO GERAL DO MERCADO:**
   - Situação atual dos setores representados pelos ativos
   - Tendências macroeconômicas relevantes
   - Sentimento geral do mercado

2. **ANÁLISE POR ATIVO:**
   - Perspectivas individuais para cada ticker (2-3 pontos-chave por ativo)
   - Fatores fundamentais e técnicos relevantes
   - Catalisadores positivos e riscos específicos

3. **CORRELAÇÕES SETORIAIS:**
   - Como estes ativos se relacionam entre si
   - Diversificação setorial e geográfica
   - Exposições comuns a riscos sistêmicos

4. **OUTLOOK E PROJEÇÕES:**
   - Perspectivas de curto prazo (1-3 meses)
   - Tendências de médio prazo (6-12 meses)
   - Temas estruturais de longo prazo

5. **FATORES DE RISCO:**
   - Principais riscos a monitorar
   - Cenários adversos e seus impactos
   - Indicadores antecedentes importantes

6. **RECOMENDAÇÕES TÁTICAS:**
   - Sugestões de posicionamento
   - Estratégias de entrada/saída
   - Níveis técnicos relevantes

Mantenha a análise atual, concisa e focada em insights acionáveis para gestores de portfólio.
Data de referência: {datetime.now().strftime('%Y-%m-%d')}
"""
            
            response = self.model.generate_content(prompt)
            
            return f"📰 **COMENTÁRIO DE MERCADO - GEMINI AI**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ Erro ao gerar comentário: {str(e)}"
    
    def generate_risk_assessment(self, tickers: List[str], portfolio_data: Dict) -> str:
        """
        Gera avaliação de risco específica do portfólio.
        
        Args:
            tickers: Lista de tickers
            portfolio_data: Dados do portfólio
            
        Returns:
            Avaliação de risco detalhada
        """
        if not self.model:
            return "❌ Google AI Studio não disponível - configure a API key"
        
        try:
            formatted_data = self._format_financial_data(portfolio_data)
            
            prompt = f"""
Como especialista em gestão de risco de portfólios, conduza uma avaliação abrangente de risco para o seguinte portfólio:

ATIVOS: {', '.join(tickers)}

DADOS DO PORTFÓLIO:
{formatted_data}

Forneça uma análise detalhada de risco incluindo:

1. **CLASSIFICAÇÃO DE RISCO:**
   - Perfil de risco geral do portfólio (conservador/moderado/agressivo)
   - Identificação dos principais fatores de risco
   - Comparação com benchmarks relevantes

2. **ANÁLISE DE CONCENTRAÇÃO:**
   - Concentração setorial, geográfica e por ativo
   - Riscos de concentração e suas implicações
   - Recomendações para diversificação

3. **STRESS TESTING:**
   - Como o portfólio reagiria a cenários adversos
   - Simulação de crises históricas (2008, 2020, etc.)
   - Identificação de pontos de vulnerabilidade

4. **VAR E MÉTRICAS DE RISCO:**
   - Estimativas de Value at Risk (VaR)
   - Expected Shortfall e máximo drawdown esperado
   - Volatilidade e beta do portfólio

5. **RECOMENDAÇÕES DE HEDGE:**
   - Estratégias específicas de proteção
   - Instrumentos de hedge recomendados
   - Custos vs benefícios das estratégias de proteção

6. **PLANO DE CONTINGÊNCIA:**
   - Protocolos para diferentes cenários de risco
   - Níveis de stop-loss e take-profit
   - Estratégias de redução de exposição

Seja específico e quantitativo sempre que possível, fornecendo recomendações práticas e implementáveis.
"""
            
            response = self.model.generate_content(prompt)
            
            return f"⚠️ **AVALIAÇÃO DE RISCO - GEMINI AI**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ Erro ao gerar avaliação de risco: {str(e)}"
    
    def set_api_key(self, api_key: str) -> bool:
        """
        Define nova API key.
        
        Args:
            api_key: Nova chave da API
            
        Returns:
            True se configurada com sucesso
        """
        try:
            self.api_key = api_key
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print("✅ Nova API key do Google AI Studio configurada com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro ao configurar API key: {e}")
            self.model = None
            return False


# Função de conveniência
def create_gemini_insights(api_key: Optional[str] = None) -> GeminiInsights:
    """
    Cria uma instância do GeminiInsights.
    
    Args:
        api_key: Chave da API Google AI Studio
        
    Returns:
        Instância configurada
    """
    return GeminiInsights(api_key)


def example_usage():
    """Exemplo de uso do módulo."""
    print("🤖 Exemplo de uso - Gemini AI Insights")
    
    # Criar instância (necessária API key)
    gemini = create_gemini_insights()  # ou GeminiInsights(api_key="sua_chave_aqui")
    
    if not gemini.model:
        print("❌ Configure sua API key do Google AI Studio primeiro!")
        print("💡 Obtenha sua chave em: https://makersuite.google.com/app/apikey")
        return
    
    # Exemplo de comentário de mercado
    tickers = ["AAPL", "GOOGL", "MSFT"]
    commentary = gemini.generate_market_commentary(tickers, "Análise para portfólio de tecnologia")
    print(commentary)


if __name__ == "__main__":
    example_usage()
