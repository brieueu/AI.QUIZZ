"""
Módulo de Insights com ChatGPT
=============================

Este módulo integra a API do OpenAI ChatGPT para fornecer análises inteligentes
e insights sobre dados financeiros, matrizes V-Cov e ponderação alfa.

Author: Gabriel
Date: 2025-09-10
"""

import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class GPTInsights:
    """
    Classe para gerar insights financeiros usando ChatGPT.
    
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
            api_key: Chave da API OpenAI (se não fornecida, tentará ler da env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("⚠️ API Key do OpenAI não encontrada. Configure OPENAI_API_KEY ou forneça a chave.")
            self.client = None
        else:
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                print("✅ Conexão com ChatGPT estabelecida!")
            except Exception as e:
                print(f"❌ Erro ao conectar com OpenAI: {e}")
                self.client = None
    
    def _format_financial_data(self, data: Dict) -> str:
        """
        Formata dados financeiros para envio ao ChatGPT.
        
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
        Analisa matriz de variância-covariância usando ChatGPT.
        
        Args:
            vcov_matrix: Matriz V-Cov
            tickers: Lista de tickers
            additional_data: Dados adicionais para contexto
            
        Returns:
            Análise textual do ChatGPT
        """
        if not self.client:
            return "❌ ChatGPT não disponível - configure a API key"
        
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
Como especialista em análise quantitativa de mercados financeiros, analise os seguintes dados de uma matriz de variância-covariância:

{formatted_data}

Forneça uma análise detalhada incluindo:

1. **ANÁLISE DE RISCO:**
   - Interpretação das volatilidades dos ativos
   - Identificação dos ativos mais e menos arriscados
   - Avaliação do nível geral de risco do portfólio

2. **ANÁLISE DE CORRELAÇÃO:**
   - Padrões de correlação entre ativos
   - Identificação de correlações altas/baixas significativas
   - Implicações para diversificação

3. **INSIGHTS ESTRATÉGICOS:**
   - Recomendações para construção de portfólio
   - Estratégias de hedge baseadas nas correlações
   - Considerações sobre timing de mercado

4. **ALERTAS E RISCOS:**
   - Potenciais armadilhas ou concentrações de risco
   - Cenários de stress que poderiam afetar o portfólio
   - Recomendações de monitoramento

Seja específico, prático e foque em ações concretas que um gestor de portfólio poderia tomar.
"""
            
            # Chamar API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise quantitativa de mercados financeiros com 15+ anos de experiência em gestão de portfólios e análise de risco."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return f"🤖 **ANÁLISE CHATGPT - MATRIZ V-COV**\n\n{response.choices[0].message.content}"
            
        except Exception as e:
            return f"❌ Erro ao gerar insights: {str(e)}"
    
    def analyze_alpha_portfolio(self, alpha_result: Dict) -> str:
        """
        Analisa resultados de ponderação alfa usando ChatGPT.
        
        Args:
            alpha_result: Resultado da análise alfa
            
        Returns:
            Análise textual do ChatGPT
        """
        if not self.client:
            return "❌ ChatGPT não disponível - configure a API key"
        
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
Como especialista em gestão quantitativa de portfólios, analise os seguintes resultados de ponderação baseada em alfa:

{formatted_data}

Forneça uma análise abrangente incluindo:

1. **ANÁLISE DE PERFORMANCE:**
   - Interpretação dos valores de alfa e beta
   - Identificação de ativos outperformers/underperformers
   - Avaliação da qualidade geral do portfólio

2. **ANÁLISE DE PONDERAÇÃO:**
   - Adequação dos pesos calculados
   - Concentração vs diversificação
   - Riscos da estratégia de ponderação

3. **ESTRATÉGIAS RECOMENDADAS:**
   - Como implementar esta ponderação na prática
   - Estratégias long/short baseadas nos alfas
   - Frequência ideal de rebalanceamento

4. **CONSIDERAÇÕES DE RISCO:**
   - Limitações da análise de alfa
   - Riscos específicos dos ativos identificados
   - Hedging e proteção de portfólio

5. **AÇÕES PRÁTICAS:**
   - Próximos passos para implementação
   - Métricas para monitoramento
   - Sinais para revisão da estratégia

Seja específico sobre implementação prática e gestão de risco.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um gestor quantitativo sênior especializado em estratégias alpha-driven e otimização de portfólios institucionais."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return f"🤖 **ANÁLISE CHATGPT - PONDERAÇÃO ALFA**\n\n{response.choices[0].message.content}"
            
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
        if not self.client:
            return "❌ ChatGPT não disponível - configure a API key"
        
        try:
            prompt = f"""
Como analista de mercados financeiros, forneça um comentário atual sobre os seguintes ativos: {', '.join(tickers)}

{f"Contexto adicional: {context}" if context else ""}

Inclua:
1. **VISÃO GERAL DO MERCADO:** Situação atual dos setores representados
2. **ANÁLISE POR ATIVO:** Perspectivas individuais (máximo 2-3 linhas por ativo)
3. **CORRELAÇÕES SETORIAIS:** Como estes ativos se relacionam
4. **OUTLOOK:** Perspectivas de curto e médio prazo
5. **FATORES DE RISCO:** Principais riscos a monitorar

Mantenha a análise concisa e focada em insights acionáveis.
Data de referência: {datetime.now().strftime('%Y-%m-%d')}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um analista de mercados financeiros com expertise em análise fundamentalista e técnica."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.8
            )
            
            return f"📰 **COMENTÁRIO DE MERCADO - CHATGPT**\n\n{response.choices[0].message.content}"
            
        except Exception as e:
            return f"❌ Erro ao gerar comentário: {str(e)}"
    
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
            self.client = openai.OpenAI(api_key=api_key)
            print("✅ Nova API key configurada com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro ao configurar API key: {e}")
            self.client = None
            return False


# Função de conveniência
def create_gpt_insights(api_key: Optional[str] = None) -> GPTInsights:
    """
    Cria uma instância do GPTInsights.
    
    Args:
        api_key: Chave da API OpenAI
        
    Returns:
        Instância configurada
    """
    return GPTInsights(api_key)


def example_usage():
    """Exemplo de uso do módulo."""
    print("🤖 Exemplo de uso - GPT Insights")
    
    # Criar instância (necessária API key)
    gpt = create_gpt_insights()  # ou GPTInsights(api_key="sua_chave_aqui")
    
    if not gpt.client:
        print("❌ Configure sua API key do OpenAI primeiro!")
        return
    
    # Exemplo de comentário de mercado
    tickers = ["AAPL", "GOOGL", "MSFT"]
    commentary = gpt.generate_market_commentary(tickers, "Análise para portfólio de tecnologia")
    print(commentary)


if __name__ == "__main__":
    example_usage()
